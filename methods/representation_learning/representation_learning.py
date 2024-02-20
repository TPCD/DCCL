import argparse
from cmath import exp
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from model import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

import torch.nn as nn
from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, km_label_path, subset_len_path, dino_pretrain_path, moco_pretrain_path, mae_pretrain_path

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data.data_utils import MergedDataset
from copy import deepcopy

from project_utils.k_means_utils import test_kmeans_semi_sup

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args):

    b_ = 0.5 * int(features.size(0))  # features:[B*2/expert_num] , b_: B/expert_num

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)  # [2*b_]
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2*b_, 2*b_]
    labels = labels.to(device)

    if args.negative_mixup:
        labels = torch.cat([torch.arange(b_ * 2) for i in range(args.n_views)], dim=0)  # [4*b_]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()   # [4*b_, 4*b_]
        labels = labels.to(device)

        beta = np.random.beta(0.2, 0.2)
        feat_idx = torch.arange(features.shape[0]-1, -1, -1)
        inter_feat = beta * features.detach().clone() + (1-beta) * features[feat_idx].detach().clone()
        inter_feat = F.normalize(inter_feat, dim=1) # [b_*2, c]
        features = torch.cat([inter_feat, features]) 

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)  

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) 
    labels = labels[~mask].view(labels.shape[0], -1) 
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) 
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) 

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(projectors, student, train_loaders, test_loader, unlabelled_train_loader, whole_train_test_loader, args):

    optimizer = SGD(list(projectors.parameters()) + list(student.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()
    best_test_acc_lab = 0
    best_acc_lab = 0

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        student.train()
        projectors.train()

        loaders =[iter(l) for l in train_loaders]

        max_len  = max([len(loader) for loader in loaders])

        # Load data from each dataloaders, the total number of dataloader is expert_num + 1
        for _ in tqdm(range(max_len)):
            fine_loss = 0
            expert_images = []
            expert_class_labels = []
            expert_mask_lab = []
            
            for idx, loader in enumerate(loaders):
                try:
                    item = next(loader)
                except StopIteration:
                    loaders[idx] = iter(train_loaders[idx])
                    item = next(loaders[idx])

                images, class_labels, uq_idxs, mask_lab = item
                mask_lab = mask_lab[:, 0]
                images = torch.cat(images, dim=0)   # [B*2/num_experts,3,224,224]
                if args.use_global_con:
                    # Load subset data
                    if idx < args.experts_num:
                        expert_images.append(images)
                        expert_class_labels.append(class_labels)
                        expert_mask_lab.append(mask_lab.bool())

                    # Load whole dataset
                    else:
                        all_images = images.to(device)
                        all_class_labels = class_labels.to(device)
                        all_mask_lab = mask_lab.to(device).bool()
                else:
                    expert_images.append(images)
                    expert_class_labels.append(class_labels)
                    expert_mask_lab.append(mask_lab.bool())

            # Cat the expert images together
            expert_images = torch.cat(expert_images, dim=0).to(device)  # [B*2,3,224,224]
            expert_class_labels = torch.cat(expert_class_labels, dim=0).to(device) 
            expert_mask_lab = torch.cat(expert_mask_lab, dim=0).to(device)

            # Extract expert features with base model
            features = student(expert_images)    # [B*2, 768]

            # Split for expert_num projectors
            features = torch.split(features, (args.batch_size//args.experts_num)*2, dim=0)
            expert_class_labels_split = torch.split(expert_class_labels, args.batch_size//args.experts_num, dim=0) 
            expert_mask_lab = torch.split(expert_mask_lab, args.batch_size//args.experts_num, dim=0) 

            if args.use_global_con:
                # Extract features of the whole train dateset
                global_features = student(all_images)


            for proj_idx, (feature, expert_class_label, expert_ml) in enumerate(zip(features, expert_class_labels_split, expert_mask_lab)): 

                projector = projectors[proj_idx]
                # Pass features through projection head
                proj_feature = projector(feature)  # [B*2/num_expert, 65536]

                # L2-normalize features
                proj_feature = torch.nn.functional.normalize(proj_feature, dim=-1)

                # Choose which instances to run the contrastive loss on
                if args.contrast_unlabel_only:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~expert_ml] for f in proj_feature.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = proj_feature

                contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[expert_ml] for f in proj_feature.chunk(2)] 
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = expert_class_label[expert_ml]

                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                # Each expert fine-grained loss
                loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss
                fine_loss +=loss
            
            fine_loss = fine_loss / args.experts_num

            if args.use_global_con:
                global_projector = projectors[args.experts_num]

                # Pass features through projection head
                global_proj_feature = global_projector(global_features)

                # L2-normalize features
                global_proj_feature = torch.nn.functional.normalize(global_proj_feature, dim=-1)

                # Choose which instances to run the contrastive loss on
                if args.contrast_unlabel_only:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~all_mask_lab] for f in global_proj_feature.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = global_proj_feature

                contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[all_mask_lab] for f in global_proj_feature.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = all_class_labels[all_mask_lab]

                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                # coarse-grained loss
                coarse_loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

                total_loss = coarse_loss + args.expert_weight * fine_loss
                # total_loss = (coarse_loss + fine_loss) / 2
            else:
                total_loss = fine_loss

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_record.update(total_loss.item(), expert_class_labels.size(0))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        print('Train Epoch: {}  Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
                                                                                  train_acc_record.avg))

        if (epoch % args.val_epoch_size == 0 or epoch == args.epochs-1):
            with torch.no_grad():

                if args.val_ssk == True:
                    print('Performing SS-K-Means on all in the training data...')
                    all_acc, old_acc, new_acc, kmeans = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                        epoch=epoch, save_name='Train ACC SSK',
                                                        args=args)
                else:
                    print('Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc = test_kmeans(student, unlabelled_train_loader,
                                                        epoch=epoch, save_name='Train ACC Unlabelled',
                                                        args=args)

                    print('Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans(student, test_loader,
                                                                    epoch=epoch, save_name='Test ACC',
                                                                    args=args)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        if (epoch % args.val_epoch_size == 0 or epoch == args.epochs-1):
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                              new_acc))
            if args.val_ssk == False:
                print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        torch.save(student.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))

        for proj_idx, projector in enumerate(projectors):
            torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_{proj_idx}.pt')
            print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_{proj_idx}.pt'))
        
        if args.val_ssk == True:
            if args.best_new == True:
                if new_acc > best_acc_lab:

                    print(f'Best ACC on new Classes on train set: {new_acc:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in enumerate(projectors):
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_acc_lab = new_acc
            else: 
                if old_acc > best_acc_lab:

                    print(f'Best ACC on old Classes on train set: {old_acc:.4f}...')
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in enumerate(projectors):
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_acc_lab = old_acc
        else:
            if old_acc_test > best_test_acc_lab:

                print(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
                print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                    new_acc))

                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                for proj_idx, projector in enumerate(projectors):
                    torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                    print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                best_test_acc_lab = old_acc_test


def test_kmeans(model, test_loader,
                epoch, save_name,
                args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":
    '''     --dataset_name 'cub' \
            --batch_size 256 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --val_epoch_size 10 \
            --use_best_model 'True' \
            --use_global_con 'True' \
            --expert_weight 0.1 \
            --max_kmeans_iter 200 \
            --k_means_init 100 \
            --best_new 'False' \
            --pretrain_model 'dino' '''

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    
    parser.add_argument('--val_epoch_size', type=int, default=50)
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--use_global_con', type=str2bool, default=True)
    parser.add_argument('--experts_num', type=int, default=8)
    parser.add_argument('--expert_weight', type=float, default=0.1)

    parser.add_argument('--val_ssk', type=str2bool, default=True)
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=200)
    parser.add_argument('--k_means_init', type=int, default=10)

    # parser.add_argument('--best_new', type=str2bool, default=False)
    parser.add_argument('--best_new', type=str2bool, default=False)

    parser.add_argument('--negative_mixup', action='store_true')
    parser.add_argument('--mixup_beta', default=0.2, type=float)
    parser.add_argument('--pretrain_model', type=str, default='dino')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print(args)
    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL PARAMS
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            pretrain_path = dino_pretrain_path
        elif args.pretrain_model == 'moco':
            pretrain_path = moco_pretrain_path
        elif args.pretrain_model == 'mae':
            pretrain_path = mae_pretrain_path
        else:
            raise NotImplementedError

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

    else:

        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_train_examples = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    whole_train_test_dataset = MergedDataset(deepcopy(labelled_train_examples),deepcopy(unlabelled_train_examples_test))

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    # train_loader: len(imgs)=2
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    if args.dataset_name == 'imagenet_100':
        test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=True)
    else:
        test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    
    train_loader_labelled = DataLoader(labelled_train_examples, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    whole_train_test_loader = DataLoader(whole_train_test_dataset,num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # Load k-means labels
    # ----------------------
    print('Loading kmeans labels...')
    subset_km_labels = np.load(f'{km_label_path}/{args.dataset_name}_km_labels.npy')
    experts_num = np.unique(subset_km_labels)
    if len(experts_num) !=args.experts_num:
        raise NotImplementedError

    # Load the length of subsets
    subset_len = []
    f = open(f'{subset_len_path}/{args.dataset_name}_subset_len.txt', 'r')
    lines = f.readlines()
    for line in lines:
        subset_len.append(int(line.strip('\n')))
    subsets_label_len = subset_len[:args.experts_num]
    subsets_unlabel_len = subset_len[args.experts_num:]

    # ----------------------
    # init model
    # ----------------------
    if args.base_model == 'vit_dino':
        student = vits.__dict__['vit_base']()

        if args.pretrain_model == 'dino':
            weight = torch.load(pretrain_path, map_location='cpu')
        elif args.pretrain_model == 'moco':
            state_dict = torch.load(pretrain_path, map_location='cpu')
            weight = {k.replace("module.base_encoder.", ""): v for k, v in state_dict['state_dict'].items()}
        elif args.pretrain_model == 'mae':
            state_dict = torch.load(pretrain_path, map_location='cpu')
            weight = state_dict['model']
        
        msg = student.load_state_dict(weight, strict=False)
        print(msg)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            student.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        student.to(device)

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in student.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in student.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError

    # ----------------------
    # projectors modulelist
    # ---------------------- 
    if args.use_global_con:
        projectors = nn.ModuleList([vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers) for _ in range(args.experts_num + 1)]).to(device)
    else:
        projectors = nn.ModuleList([vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers) for _ in (experts_num)]).to(device)
    # ----------------------
    # Building sub dataset
    # -------------------
    print('Building sub dataset...')
    sub_data_loaders = []
    sub_num = []
    for subset_index in experts_num:
        # Get subset index
        subset_idx = []
        for idx, km_label in enumerate(subset_km_labels):
            if km_label == subset_index:
                subset_idx.append(idx)
        
        # Get the subset
        subset = torch.utils.data.Subset(deepcopy(train_dataset), subset_idx)  # len(subset_unlabelled)=, len(datasets['train_unlabelled'])=4496

        # Get the numbers of each subset
        subdata_num = np.sum(subset_km_labels == subset_index)
        sub_num.append(subdata_num)

        sub_label_len = subsets_label_len[subset_index]
        sub_unlabelled_len = subsets_unlabel_len[subset_index]
        sub_sample_weights = [1 if i < sub_label_len else sub_label_len / sub_unlabelled_len for i in range(len(subset))]
        sub_sample_weights = torch.DoubleTensor(sub_sample_weights)
        sub_sampler = torch.utils.data.WeightedRandomSampler(sub_sample_weights, num_samples=len(subset))

        # Dala loader for subset
        sub_data_loader = DataLoader(subset, num_workers=args.num_workers,
                                        batch_size=args.batch_size//args.experts_num, sampler=sub_sampler, shuffle=False, drop_last=True)

        sub_data_loaders.append(sub_data_loader)
    
    if args.use_global_con:
        # Train loaders have 8 subsets + 1 whole set
        train_loaders = sub_data_loaders + [train_loader]
    else:
        train_loaders = sub_data_loaders
    
    print('Length of sub dataset: ', sub_num)

    # Train student model
    train(projectors, student, train_loaders, test_loader, test_loader_unlabelled, whole_train_test_loader, args)