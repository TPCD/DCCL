import argparse
import os
from easydict import EasyDict

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
from data.data_utils import MergedDataset

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, km_label_path, dino_pretrain_path, moco_pretrain_path, mae_pretrain_path

from copy import deepcopy
# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--experts_num', type=int, default=8)

    parser.add_argument('--pretrain_model', type=str, default='dino')

    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            pretrain_path = dino_pretrain_path

        model = vits.__dict__['vit_base']()

        if args.pretrain_model == 'dino':
            weight = torch.load(pretrain_path, map_location='cpu')
        
        msg = model.load_state_dict(weight, strict=False)
        print(msg)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:

        raise NotImplementedError

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
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    train_loader_labelled = DataLoader(labelled_train_examples, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)
    whole_train_test_loader = DataLoader(whole_train_test_dataset, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_feats = []
    targets = np.array([])
    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(whole_train_test_loader)):
        images, label, _, _ = batch
        images = images.to(device)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
    print('Kmeans...')

    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.experts_num, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')
    print('feats length:', all_feats.shape)

    np.save(f'{km_label_path}/{args.dataset_name}_km_labels.npy', preds)
