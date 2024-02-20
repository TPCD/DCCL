import argparse
from cmath import exp
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler, Adam
from project_utils.cluster_utils import mixed_eval, AverageMeter
from model import vision_transformer as vits
from model import attribute_transformer as ats
import higher
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, km_label_path, subset_len_path, dino_base_pretrain_path, dino_small_pretrain_path, cub_root
import time
from data.data_utils import MergedDataset
from copy import deepcopy
from torch.cuda.amp import autocast as autocast
from project_utils.k_means_utils import test_kmeans_semi_sup, test_kmeans
from project_utils.cluster_and_log_utils import Logger
from warmup_scheduler import GradualWarmupScheduler
from project_utils.contrastive_utils import extract_features, accuracy, info_nce_logits, \
    ContrastiveLearningViewGenerator, SupConLoss
from project_utils.infomap_cluster_utils import cluster_by_semi_infomap, get_dist_nbr, cluster_by_infomap, generate_cluster_features
from project_utils.cluster_memory_utils import ClusterMemory
from project_utils.data_utils import IterLoader, FakeLabelDataset
from collections import defaultdict
from project_utils.sampler import RandomMultipleGallerySamplerNoCam
# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# generate new dataset and calculate cluster centers

global_seed = 2022
random_seed = False
if random_seed:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.determinstic = False
else:
    set_seed(global_seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def train(projectors, student, train_loaders_dict, test_loader, unlabelled_train_loader, whole_train_test_loader,
          ssk_test_loader, args,
          meta_learner=None):
    backbone_optimizer = SGD(student.parameters(), lr=args.backbone_lr, momentum=args.momentum,
                             weight_decay=args.weight_decay)
    head_optimizer = SGD(projectors.parameters(), lr=args.head_lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    if meta_learner is not None:
        meta_optimizer = Adam(meta_learner.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)

    backbone_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        backbone_optimizer,
        T_max=args.epochs,
        eta_min=args.backbone_lr * args.backbone_eta_min,
    )
    backbone_scheduler_warmup = GradualWarmupScheduler(backbone_optimizer, multiplier=1,
                                                       total_epoch=args.num_warmup_epoch,
                                                       after_scheduler=backbone_lr_scheduler)

    head_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        head_optimizer,
        T_max=args.epochs,
        eta_min=args.head_lr * args.head_eta_min,
    )

    contrastive_cluster_weight_schedule = np.concatenate((
        np.linspace(args.contrastive_cluster_weight * 0.1,
                    args.contrastive_cluster_weight, args.contrastive_cluster_epochs),
        np.ones(args.epochs - args.contrastive_cluster_epochs) * args.contrastive_cluster_weight
    ))


    head_scheduler_warmup = GradualWarmupScheduler(head_optimizer, multiplier=1, total_epoch=args.num_warmup_epoch,
                                                   after_scheduler=head_lr_scheduler)
    best_text_acc_epoch = 0
    best_test_acc_lab = 0
    best_acc_lab = 0
    contrastive_cluster_train_loader_predefine = None
    for epoch in range(args.epochs):
        # test for valudate pretrain parameter
        if args.test_before_train:
            if epoch == 0:
                with torch.no_grad():
                    if 'kmeans' in args.val_ssk:
                        logger('Performing general K-Means: Testing on unlabeled train set...')
                        all_acc_test, old_acc_test, new_acc_test = test_kmeans(student, unlabelled_train_loader,
                                                                               epoch=epoch, save_name='Test ACC',
                                                                               device=device,
                                                                               args=args, logger_class=logger)
                        logger('Disjoint unlabeled train set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                            all_acc_test, old_acc_test,
                            new_acc_test))

                    if 'ssk' in args.val_ssk:
                        logger('Performing SS-K-Means: Testing on both labelled and unlabelled training data...')
                        all_acc, old_acc, new_acc, kmeans, all_feature = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                                                 epoch=epoch,
                                                                                 save_name='Train ACC SSK ALL',
                                                                                 device=device,
                                                                                 args=args, logger_class=logger, in_training=True)
                        logger('SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                              new_acc))


        else:
            old_acc_test = 0.0
            new_acc_test = 0.0
            all_acc_test = 0.0
            all_acc, old_acc, new_acc = 0.0, 0.0, 0.0

        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        iterator_dict = {k: iter(v) for k, v in train_loaders_dict.items()}

        if args.use_contrastive_cluster:
            if args.contrastive_cluster_method == 'infomap' and epoch % args.epoch_per_clustering == 0:
                with torch.no_grad():
                    print('==> Create pseudo labels for unlabeled data')
                    # cluster_loader = unlabelled_train_loader
                    cluster_loader = deepcopy(whole_train_test_loader)
                    features, labels, if_labeled = extract_features(student, cluster_loader, print_freq=50, args=args)
                    features = torch.cat(features, dim=0)
                    label_mark = torch.cat(labels, dim=0)
                    if_labeled = torch.cat(if_labeled, dim=0)
                    # features = torch.cat([features[f].unsqueeze(0) for f, _, _ in unlabelled_train_examples_test.data], 0)
                    features_array = F.normalize(features, dim=1).cpu().numpy()
                    feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu', device=GPU_INDEX)
                    del features_array

                    s = time.time()
                    pseudo_labels = cluster_by_semi_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2, label_mark=label_mark, if_labeled=if_labeled, args=args)
                    pseudo_labels = pseudo_labels.astype(np.intp)

                    print('cluster cost time: {}'.format(time.time() - s))
                    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

                cluster_features = generate_cluster_features(pseudo_labels, features)

                del cluster_loader, features

                # Create hybrid memory
                memory = ClusterMemory(student.num_features, num_cluster, temp=args.temp,
                                       momentum=args.memory_momentum, use_hard=args.use_hard).to(args.device)

                memory.features = F.normalize(cluster_features, dim=1).to(args.device)
                # trainer.memory = memory
                pseudo_labeled_dataset = []

                for i, (_item, label) in enumerate(zip(whole_train_test_dataset.data, pseudo_labels)):
                    if label != -1:
                        if isinstance(_item, str):
                            pseudo_labeled_dataset.append((_item, label.item(), _item))
                        elif args.dataset_name == 'imagenet_100':
                            pseudo_labeled_dataset.append((_item[0], label.item(), _item[1]))
                        else:
                            pseudo_labeled_dataset.append((_item[1], label.item(), _item[2]))

                logger('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

                # train_loader = get_train_loader(args, dataset, args.height, args.width,
                #                                 args.batch_size, args.workers, args.num_instances, iters,
                #                                 trainset=pseudo_labeled_dataset, no_cam=args.no_cam)
                PK_sampler = RandomMultipleGallerySamplerNoCam(pseudo_labeled_dataset, args.num_instances)

                # image_dir = os.path.join(unlabelled_train_examples_test.root, unlabelled_train_examples_test.base_folder)
                contrastive_cluster_train_loader = IterLoader(
                    DataLoader(FakeLabelDataset(pseudo_labeled_dataset, root=None, transform=train_transform),
                               batch_size=args.batch_size, num_workers=args.num_workers, sampler=PK_sampler,
                               shuffle=False, pin_memory=True, drop_last=True))
                contrastive_cluster_train_loader.new_epoch()
                contrastive_cluster_train_loader_predefine = contrastive_cluster_train_loader
            elif args.contrastive_cluster_method == 'ssk':
                if epoch % args.epoch_per_clustering == 0:
                    with torch.no_grad():
                        logger('==> Create pseudo labels for unlabeled data by ssk!')
                        logger('Performing SS-K-Means: Testing on both labelled and unlabelled training data...')
                        s = time.time()
                        all_acc, old_acc, new_acc, kmeans, all_feats = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                                                epoch=epoch,
                                                                                save_name='Train ACC SSK ALL',
                                                                                device=device,
                                                                                args=args, logger_class=logger, in_training=True)
                        logger('SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                              new_acc))
                        pseudo_labels = kmeans.labels_.cpu().numpy()
                        features = torch.from_numpy(all_feats).to(args.device)

                        print('cluster cost time: {}'.format(time.time() - s))
                        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

                    cluster_features = generate_cluster_features(pseudo_labels, features)

                    del features

                    # Create hybrid memory
                    memory = ClusterMemory(student.num_features, num_cluster, temp=args.temp,
                                           momentum=args.memory_momentum, use_hard=args.use_hard).to(args.device)

                    memory.features = F.normalize(cluster_features, dim=1).to(args.device)
                    # trainer.memory = memory
                    pseudo_labeled_dataset = []

                    for i, (_item, label) in enumerate(zip(whole_train_test_dataset.data, pseudo_labels)):
                        if label != -1:
                            if isinstance(_item, str):
                                pseudo_labeled_dataset.append((_item, label.item(), _item))
                            else:
                                pseudo_labeled_dataset.append((_item[1], label.item(), _item[2]))

                    logger('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

                    # train_loader = get_train_loader(args, dataset, args.height, args.width,
                    #                                 args.batch_size, args.workers, args.num_instances, iters,
                    #                                 trainset=pseudo_labeled_dataset, no_cam=args.no_cam)
                    PK_sampler = RandomMultipleGallerySamplerNoCam(pseudo_labeled_dataset, args.num_instances)

                    # image_dir = os.path.join(unlabelled_train_examples_test.root, unlabelled_train_examples_test.base_folder)
                    contrastive_cluster_train_loader = IterLoader(
                        DataLoader(FakeLabelDataset(pseudo_labeled_dataset, root=None, transform=train_transform),
                                   batch_size=args.batch_size, num_workers=args.num_workers, sampler=PK_sampler,
                                   shuffle=False, pin_memory=True, drop_last=True))
                    contrastive_cluster_train_loader.new_epoch()
                    contrastive_cluster_train_loader_predefine = contrastive_cluster_train_loader

        # gcd learning
        student.train()
        projectors.train()

        max_len = max([len(iterator) for iterator in iterator_dict.values()])

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        # Load data from each dataloaders, the total number of dataloader is expert_num + 1 + 1
        for iteration in tqdm(range(max_len)):
            global_contrastive_dict, attribute_dict = {}, {}
            base_total_loss = 0.0
            try:
                item = next(iterator_dict['global_contrastive'])
            except StopIteration:
                iterator_dict['global_contrastive'] = iter(iterator_dict['global_contrastive'])
                item = next(iterator_dict['global_contrastive'])
            # images 16 per expert * 2 view
            images, class_labels, uq_idxs, label_masks, attributes = item
            class_labels, uq_idxs, label_masks, attributes = class_labels.to(device), uq_idxs.to(
                device), label_masks.to(device), attributes.to(device)
            label_masks = label_masks[:, 0]
            if isinstance(images, (list, tuple)):  # concat multiview
                images = torch.cat(images, dim=0).to(device)  # [B*2/num_experts,3,224,224]
            else:
                images = images.to(device)

            global_contrastive_dict['images'] = images
            global_contrastive_dict['class_labels'] = class_labels
            global_contrastive_dict['label_masks'] = label_masks.bool()
            global_contrastive_dict['attributes'] = attributes

            # Extract features of the whole train dateset
            tuple_output = student(global_contrastive_dict['images'])
            global_features = tuple_output[0]
            attribute_features = tuple_output[1]
            meta_features = tuple_output[2]

            global_projector = projectors['global_contrastive']

            # Pass features through projection head
            global_proj_feature = global_projector(global_features)

            # L2-normalize features
            global_proj_feature = torch.nn.functional.normalize(global_proj_feature, dim=-1)

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~global_contrastive_dict['label_masks']] for f in global_proj_feature.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = global_proj_feature

            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # Supervised contrastive loss
            f1, f2 = [f[global_contrastive_dict['label_masks']] for f in global_proj_feature.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = global_contrastive_dict['class_labels'][global_contrastive_dict['label_masks']]

            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # coarse-grained loss
            global_contrastive_loss = (
                                              1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss
            # print(f'before scale global_contrastive_loss: {global_contrastive_loss.item()}')
            # global_contrastive_loss = scaler.scale(global_contrastive_loss)
            base_total_loss += global_contrastive_loss
            if args.use_attribute:
                try:
                    item = next(iterator_dict['attribute'])
                except StopIteration:
                    iterator_dict['attribute'] = iter(iterator_dict['attribute'])
                    item = next(iterator_dict['attribute'])
                # images 16 per expert * 2 view
                images, class_labels, uq_idxs, label_masks, attributes = item
                class_labels, uq_idxs, label_masks, attributes = class_labels.to(device), uq_idxs.to(
                    device), label_masks.to(device), attributes.to(device)
                label_masks = label_masks[:, 0]
                if isinstance(images, (list, tuple)):  # concat multiview
                    images = torch.cat(images, dim=0).to(device)  # [B*2/num_experts,3,224,224]
                else:
                    images = images.to(device)

                attribute_dict['images'] = images
                attribute_dict['class_labels'] = class_labels
                attribute_dict['label_masks'] = label_masks.bool()
                attribute_dict['attributes'] = attributes

                _, attribute_features, _ = student(attribute_dict['images'], attribute_dict['attributes'])
                attribute_classifier = projectors['attributes']
                # Pass features through projection head
                logits_list = attribute_classifier(attribute_features)
                n_attr = len(logits_list)
                attribute_ce_loss = 0
                acc_list = []
                for attributes_id, _logit in enumerate(logits_list):
                    attribute_ce_loss += attribute_crit(_logit,
                                                        attribute_dict['attributes'][:, attributes_id].long())

                    acc_list.append(accuracy(_logit, attribute_dict['attributes'][:, attributes_id].long()))
                attribute_ce_loss = attribute_ce_loss / n_attr
                base_total_loss += attribute_ce_loss

            if args.use_contrastive_cluster:
                if args.momentum_update:
                    images, labels, indexes = contrastive_cluster_train_loader_predefine.next()
                    data_time.update(time.time() - end)
                    if isinstance(images, (list, tuple)):  # concat multiview
                        images = torch.cat(images, dim=0).to(device)  # [B*2/num_experts,3,224,224]
                        labels2 = labels.detach().clone()
                        labels = torch.cat((labels, labels2), dim=0).to(device)
                    else:
                        images = images.to(device)
                        labels = labels.to(device)

                    # forward
                    f_out, _, _ = student(images)

                    # compute loss with the hybrid memory
                    # loss = self.memory(f_out, indexes)
                    contrastive_cluster_loss = memory(f_out, labels)
                    base_total_loss += contrastive_cluster_weight_schedule[epoch] * contrastive_cluster_loss.to(args.device)
                    losses.update(contrastive_cluster_loss.item())

                    # print log
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if iteration % 20 == 0:
                        print('Epoch: [{}][{}/{}]\t'
                            'Time {:.3f} ({:.3f})\t'
                            'Data {:.3f} ({:.3f})\t'
                            'Cluster Loss {:.3f} ({:.3f})'
                            .format(epoch, iteration, max_len,
                                    batch_time.val, batch_time.avg,
                                    data_time.val, data_time.avg,
                                    losses.val, losses.avg))
            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))
            loss_record.update(base_total_loss.item(), global_contrastive_dict['class_labels'].size(0))

            backbone_optimizer.zero_grad()
            head_optimizer.zero_grad()
            base_total_loss.backward()
            backbone_optimizer.step()
            head_optimizer.step()

        if args.use_attribute:
            acc_list = np.array(acc_list)
            avrage_acc = sum(acc_list) / len(acc_list)
            logger(f'Attribute classifier average accuracy: {float(avrage_acc)} \n')
            logger([f'{float(_acc):.2f}' for _acc in acc_list])

        if ((epoch % args.val_epoch_size == 0) and (epoch > 0)) or (epoch == args.epochs - 1):
            with torch.no_grad():

                if 'kmeans' in args.val_ssk:
                    logger('Performing general K-Means: Testing on unlabelled examples in the training data...')
                    all_acc, old_acc, new_acc = test_kmeans(student, unlabelled_train_loader,
                                                            epoch=epoch, save_name='Train ACC Unlabelled',
                                                            device=device,
                                                            args=args, logger_class=logger)
                    logger('Unlabelled training set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                        all_acc, old_acc,
                        new_acc))

                    logger('Performing general K-Means: Testing on disjoint test set...')
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans(student, test_loader,
                                                                           epoch=epoch, save_name='Test ACC',
                                                                           device=device,
                                                                           args=args, logger_class=logger)
                    logger('Disjoint testing set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
                        all_acc_test, old_acc_test,
                        new_acc_test))

                if 'ssk' in args.val_ssk:
                    logger('Performing SS-K-Means: Testing on dis-joint test data...')
                    all_acc, old_acc, new_acc, kmeans, all_feature = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                                             epoch=epoch, save_name='Train ACC SSK ALL',
                                                                             device=device,
                                                                             args=args, logger_class=logger, in_training=True)
                    logger('SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                          new_acc))
        logger(
            'Train Epoch: {}  Avg Loss: {:.4f} | Seen Class Acc: {:.4f} | backbone_lr {:.6f} | head_lr {:.6f}'.format(
                epoch, loss_record.avg,
                train_acc_record.avg, get_mean_lr(backbone_optimizer), get_mean_lr(head_optimizer)))

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('Backbone_LR', get_mean_lr(backbone_optimizer), epoch)
        args.writer.add_scalar('Head_LR', get_mean_lr(head_optimizer), epoch)

        if (epoch % args.val_epoch_size == 0 or epoch == args.epochs - 1) and (epoch > 0):
            logger(
                'Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | backbone_lr {:.6f} | head_lr {:.6f}'.format(
                    all_acc, old_acc,
                    new_acc, get_mean_lr(backbone_optimizer), get_mean_lr(head_optimizer)))
            if 'ssk' not in args.val_ssk:
                logger('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                      new_acc_test))

        # Step schedule
        # backbone_lr_scheduler.step()
        backbone_scheduler_warmup.step()
        # head_lr_scheduler.step()
        head_scheduler_warmup.step()


        torch.save(student.state_dict(), args.model_path)
        logger("model saved to {}.".format(args.model_path))

        for proj_idx, projector in projectors.items():
            torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_{proj_idx}.pt')
            logger("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_{proj_idx}.pt'))

        if 'ssk' in args.val_ssk:
            if args.best_new == True:
                if new_acc > best_acc_lab:
                    best_text_acc_epoch = epoch
                    logger(f'Best ACC on new Classes on train set: {new_acc:.4f} at epoch {epoch}')
                    logger('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                                new_acc))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    logger("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in projectors.items():
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        logger("projection head saved to {}.".format(
                            args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_acc_lab = new_acc
            else:
                if old_acc > best_acc_lab:
                    best_text_acc_epoch = epoch
                    logger(f'Best ACC on old Classes on train set: {old_acc:.4f} at epoch {epoch}')
                    logger('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                                new_acc))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    logger("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in projectors.items():
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        logger("projection head saved to {}.".format(
                            args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_acc_lab = old_acc
        else:
            if args.best_new == True:
                if new_acc_test > best_test_acc_lab:
                    best_text_acc_epoch = epoch
                    logger(f'Best ACC on new Classes on disjoint test set: {new_acc_test:.4f} at epoch {epoch}')
                    logger(
                        'Best General kmeans Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test,
                                                                                                           old_acc_test,
                                                                                                           new_acc_test))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    logger("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in projectors.items():
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        logger(
                            "projection head saved to {}.".format(
                                args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_test_acc_lab = old_acc_test
            else:
                if old_acc_test > best_test_acc_lab:
                    best_text_acc_epoch = epoch
                    logger(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f} at epoch {epoch}')
                    logger('Best General kmeans Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc,
                                                                                                              old_acc,
                                                                                                              new_acc))

                    torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                    logger("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                    for proj_idx, projector in projectors.items():
                        torch.save(projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt')
                        logger(
                            "projection head saved to {}.".format(
                                args.model_path[:-3] + f'_proj_head_best_{proj_idx}.pt'))

                    best_test_acc_lab = old_acc_test

    return args.model_path[:-3] + f'_best.pt', best_text_acc_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v1'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='at13', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: imagenet_100,cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--backbone_lr', type=float, default=0.1)
    parser.add_argument('--backbone_eta_min', type=float, default=1e-3)

    parser.add_argument('--head_lr', type=float, default=0.1)
    parser.add_argument('--head_eta_min', type=float, default=1e-3)

    parser.add_argument('--meta_lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=global_seed, type=int)

    parser.add_argument('--base_model', type=str, default='base')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--val_epoch_size', type=int, default=30)
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--use_global_con', type=str2bool, default=True)
    parser.add_argument('--global_con_weight', type=float, default=1)

    # attribute branch
    parser.add_argument('--use_attribute', type=str2bool, default=False)
    parser.add_argument('--attribute_weight', type=float, default=0.1)
    parser.add_argument('--use_expert', type=bool, default=False)
    parser.add_argument('--experts_num', type=int, default=8)
    parser.add_argument('--expert_weight', type=float, default=0.1)

    parser.add_argument('--val_ssk', type=list, default=['kmeans','ssk'])
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=200)
    parser.add_argument('--train_max_kmeans_iter', type=int, default=200)
    parser.add_argument('--k_means_init', type=int, default=100)

    # parser.add_argument('--best_new', type=str2bool, default=False)
    parser.add_argument('--best_new', type=bool, default=True)

    parser.add_argument('--negative_mixup', default=False)
    parser.add_argument('--mixup_beta', default=0.2, type=float)
    parser.add_argument('--pretrain_model', type=str, default='dino')

    # meta learning
    parser.add_argument('--use_meta_attribute', type=bool, default=False)
    parser.add_argument('--begin_meta_training', type=int, default=-1)

    parser.add_argument('--test_before_train', type=bool, default=True)
    parser.add_argument('--num_warmup_epoch', type=int, default=5)

    parser.add_argument('--amp', type=bool, default=False)

    # cluster
    parser.add_argument('--use_contrastive_cluster', type=bool, default=True,
                        help="hyperparameter for KNN")
    parser.add_argument('--contrastive_cluster_method', type=str, default='infomap',
                        help="hyperparameter for KNN")
    parser.add_argument('--epoch_per_clustering', type=int, default=1,
                        help="hyperparameter for KNN")
    parser.add_argument('--momentum_update', type=bool, default=True,
                        help="hyperparameter for KNN")
    parser.add_argument('--use_l2_in_ssk', type=bool, default=True,
                        help="hyperparameter for KNN")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")

    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    parser.add_argument('--use-hard', default=False)
    parser.add_argument('--memory_momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # memory
    parser.add_argument('--use_cluster_head', type=bool, default=False,
                        help="learning rate")
    parser.add_argument('--num_instances', type=int, default=16)
    parser.add_argument('--contrastive_cluster_weight', type=float, default=0.4)
    parser.add_argument('--contrastive_cluster_epochs', type=int, default=100, help=['a-1 -> a', 'a->a'])
    parser.add_argument('--max_sim', type=bool, default=True)
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for adj")
    # ----------------------
    # INIT
    # ----------------------
    if 'G' or 'GPU' in os.path.basename(__file__).split('_')[0]:
        GPU_INDEX = int(os.path.basename(__file__).split('_')[0][-1])
    else:
        GPU_INDEX = 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)

    base_scaler = torch.cuda.amp.GradScaler()
    meta_scaler = torch.cuda.amp.GradScaler()
    device = torch.device('cuda:' + str(GPU_INDEX))
    args = parser.parse_args()
    args.device = device

    # device = torch.device('cuda')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # print(args)
    init_experiment(args, runner_name=['metric_learn_gcd'])
    logger = Logger(os.path.join(args.log_dir, 'log_out.txt'))
    logger(args)
    logger(f'Executing {os.path.basename(__file__)}, log_dir = {args.log_dir}')
    logger(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL PARAMS
    # ----------------------
    if args.base_model == 'base':

        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            pretrain_path = dino_base_pretrain_path
        else:
            raise NotImplementedError

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536
        args.feat_channal = 197  # (13+1)*(13+1) + 1 class token
        args.attribute_feat_channal = 16  # (13+1)*(13+1) + 1 class token

    elif args.base_model == 'small':
        args.interpolation = 3
        args.crop_pct = 0.875
        if args.pretrain_model == 'dino':
            pretrain_path = dino_small_pretrain_path
        else:
            raise NotImplementedError

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 384
        args.num_mlp_layers = 3
        args.mlp_out_dim = 65536
        args.feat_channal = 197  # (13+1)*(13+1) + 1 class token
        args.attribute_feat_channal = 16  # (13+1)*(13+1) + 1 class token
    else:
        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # 16 batch => [16 v1 batch, 16 v2 batch] 32 batch
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, \
    labelled_train_examples_attribute = get_datasets(args.dataset_name,
                                                     train_transform,
                                                     test_transform,
                                                     args)

    whole_train_test_dataset = MergedDataset(deepcopy(labelled_train_examples),
                                             deepcopy(unlabelled_train_examples_test))
    ssk_test_dataset = MergedDataset(deepcopy(labelled_train_examples),
                                     deepcopy(test_dataset))
    labelled_train_examples_attribute_dataset = MergedDataset(deepcopy(labelled_train_examples), None)

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
    # train_loader: len(imgs)=2 for gloabel expert train
    #
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
    labelled_train_examples_attribute_loader = DataLoader(labelled_train_examples_attribute_dataset,
                                                          num_workers=args.num_workers,
                                                          batch_size=args.batch_size, shuffle=False)
    whole_train_test_loader = DataLoader(whole_train_test_dataset, num_workers=args.num_workers,
                                         batch_size=args.batch_size, shuffle=False)

    ssk_test_loader = DataLoader(ssk_test_dataset, num_workers=args.num_workers,
                                 batch_size=args.batch_size, shuffle=False)
    # ----------------------
    # init model
    # ----------------------
    if hasattr(labelled_train_examples, 'dict_attribute'):
        num_attribute = len(labelled_train_examples.dict_attribute.keys())
    else:
        num_attribute = 28
    logger(f"num_attribute => {num_attribute}!")
    if args.base_model == 'base':
        if args.model_name == 'at':
            student = ats.__dict__['at_base'](pretrain_path)
        elif args.model_name == 'at2':
            student = ats.__dict__['at2_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at3':
            student = ats.__dict__['at3_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at4':
            student = ats.__dict__['at4_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at5':
            student = ats.__dict__['at5_base'](pretrain_path, num_attribute=num_attribute,
                                               feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at6':
            student = ats.__dict__['at6_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at7':
            student = ats.__dict__['at7_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at8':
            student = ats.__dict__['at8_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at9':
            student = ats.__dict__['at9_base'](pretrain_path, num_attribute=num_attribute,
                                               attribute_feat_channal=args.attribute_feat_channal,
                                               grad_from_block=args.grad_from_block)
        elif args.model_name == 'at10':
            student = ats.__dict__['at10_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta1_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at11':
            student = ats.__dict__['at11_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at12':
            student = ats.__dict__['at12_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at13':
            student = ats.__dict__['at13_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        elif args.model_name == 'at14':
            student = ats.__dict__['at14_base'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block, device=device)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_base'](pretrain_path, labelled_train_examples.dict_attribute,
                                                          grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        else:
            raise NotImplementedError
        # student = vits.__dict__['vit_base']()
        # student = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    elif args.base_model == 'small':
        if args.model_name == 'at':
            student = ats.__dict__['at_small'](pretrain_path)
        elif args.model_name == 'at2':
            student = ats.__dict__['at2_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at3':
            student = ats.__dict__['at3_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at4':
            student = ats.__dict__['at4_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at5':
            student = ats.__dict__['at5_small'](pretrain_path, num_attribute=num_attribute,
                                                feat_channal=args.feat_channal, grad_from_block=args.grad_from_block)
        elif args.model_name == 'at6':
            student = ats.__dict__['at6_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at7':
            student = ats.__dict__['at7_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at8':
            student = ats.__dict__['at8_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at9':
            student = ats.__dict__['at9_small'](pretrain_path, num_attribute=num_attribute,
                                                attribute_feat_channal=args.attribute_feat_channal,
                                                grad_from_block=args.grad_from_block)
        elif args.model_name == 'at10':
            student = ats.__dict__['at10_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta1_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at11':
            student = ats.__dict__['at11_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at12':
            student = ats.__dict__['at12_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        elif args.model_name == 'at13':
            student = ats.__dict__['at13_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None

        elif args.model_name == 'at14':
            student = ats.__dict__['at14_small'](pretrain_path, num_attribute=num_attribute,
                                                 attribute_feat_channal=args.attribute_feat_channal,
                                                 grad_from_block=args.grad_from_block, device=device)
            if args.use_meta_attribute:
                meta_learner = ats.__dict__['meta2_small'](pretrain_path, labelled_train_examples.dict_attribute,
                                                           grad_from_block=args.grad_from_block)
                logger(meta_learner)
                meta_learner.to(device)
            else:
                meta_learner = None
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.warmup_model_dir is not None:
        logger(f'Loading weights from {args.warmup_model_dir}')
        student.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    logger(student)
    student.to(device)

    # ----------------------
    # projectors modulelist
    # ----------------------
    projectors = nn.ModuleDict()
    if args.use_expert:
        for expert_id in range(args.experts_num):
            projectors[f'expert_{expert_id + 1}'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                                              out_dim=args.mlp_out_dim,
                                                                              nlayers=args.num_mlp_layers).to(device)
    if args.use_global_con:
        projectors['global_contrastive'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                                     out_dim=args.mlp_out_dim,
                                                                     nlayers=args.num_mlp_layers).to(device)

    if args.use_attribute:
        projectors['attributes'] = vits.__dict__['Attribute_Classifier8ind'](labelled_train_examples.dict_attribute,
                                                                             in_dim=args.feat_dim,
                                                                             ).to(device)
    if args.use_cluster_head:
        projectors['cluster'] = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                          out_dim=args.feat_dim,
                                                          nlayers=args.num_mlp_layers).to(device)

    logger(projectors)

    fake_attribute_crit = nn.KLDivLoss(reduce=True, size_average=False).to(device)
    attribute_crit = nn.CrossEntropyLoss(reduction='mean').to(device)
    sup_con_crit = SupConLoss(device).to(device)

    train_loaders_dict = {}

    # if args.use_expert:
    #     train_loaders.extend(sub_data_loaders)
    if args.use_global_con:
        train_loaders_dict['global_contrastive'] = train_loader
    if args.use_attribute:
        train_loaders_dict['attribute'] = labelled_train_examples_attribute_loader
    if args.use_contrastive_cluster:
        if args.contrastive_cluster_method == 'ssk':
            train_loaders_dict['labelled_train'] = train_loader_labelled
    # Train loaders have 1 whole set + 1 labelled attr + unlabel train

    # Train student model
    best_model_path, best_model_epoch = train(projectors, student, train_loaders_dict, test_loader,
                                              test_loader_unlabelled,
                                              whole_train_test_loader, ssk_test_loader, args,
                                              meta_learner)

    with torch.no_grad():

        logger('Use final model to test!')
        logger('Performing general K-Means: Testing on unlabeled train set...')
        all_acc_test, old_acc_test, new_acc_test = test_kmeans(student, test_loader_unlabelled,
                                                               epoch=args.epochs, save_name='Test ACC',
                                                               device=device,
                                                               args=args, logger_class=logger)
        logger('Disjoint unlabeled train set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
            all_acc_test, old_acc_test,
            new_acc_test))

        all_acc, old_acc, new_acc, kmeans, all_features = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                                               epoch=args.epochs,
                                                                               save_name='Test Best ACC SSK ALL',
                                                                               device=device,
                                                                               args=args, logger_class=logger,
                                                                               in_training=False)
        logger('Best SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                   new_acc))



        logger('Use best model to test!')
        logger('Performing general K-Means: Testing on unlabeled train set...')
        all_acc_test, old_acc_test, new_acc_test = test_kmeans(student, test_loader_unlabelled,
                                                               epoch=args.epochs, save_name='Test ACC',
                                                               device=device,
                                                               args=args, logger_class=logger)
        logger('Disjoint unlabeled train set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
            all_acc_test, old_acc_test,
            new_acc_test))

        student.load_state_dict(torch.load(best_model_path, map_location='cpu'))

        all_acc, old_acc, new_acc, kmeans, all_features = test_kmeans_semi_sup(student, whole_train_test_loader,
                                                                               epoch=args.epochs,
                                                                               save_name='Test Best ACC SSK ALL',
                                                                               device=device,
                                                                               args=args, logger_class=logger,
                                                                               in_training=False)
        logger('Best SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                   new_acc))

