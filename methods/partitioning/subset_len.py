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
from config import exp_root, km_label_path, subset_len_path, dino_pretrain_path

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
    '''${PYTHON} -m methods.partitioning.subset_len \
            --dataset_name 'cub' \
            --batch_size 256 \
            --num_workers 4 \
            --use_ssb_splits 'True' \
            --transform 'imagenet' \
            --eval_funcs 'v2' \
            --experts_num 8 \''''
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--experts_num', type=int, default=8)

    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    whole_km_labels = np.load(f'{km_label_path}/{args.dataset_name}_km_labels.npy')
    experts_num = np.unique(whole_km_labels)
    print('subset_num:', experts_num)
    if len(experts_num) !=args.experts_num:
        raise NotImplementedError

    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_train_examples = get_datasets(args.dataset_name,
                                                                                            train_transform,
                                                                                            test_transform,
                                                                                            args)
    print('whole dataset:', len(train_dataset))
    print('labelled:', len(labelled_train_examples))
    print('unlabelled:', len(unlabelled_train_examples_test))

    # ----------------------
    # Building sub dataset
    # -------------------
    sub_data_loaders = []
    sub_num = []
    for subset_index in experts_num:
        # Get subset index
        subset_idx = []
        for idx, km_label in enumerate(whole_km_labels):
            if km_label == subset_index:
                subset_idx.append(idx)

        # Get the subset
        subset = torch.utils.data.Subset(deepcopy(train_dataset), subset_idx)  # len(subset_unlabelled)=, len(datasets['train_unlabelled'])=4496

        # Get the numbers of each subset
        subdata_num = np.sum(whole_km_labels == subset_index)
        sub_num.append(subdata_num)

        # Dala loader for subset
        sub_data_loader = DataLoader(subset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
        sub_data_loaders.append(sub_data_loader)

    label_nums = []
    unlabel_nums = []
    for loader in sub_data_loaders:
        label_num = 0
        unlabel_num = 0
        for batch in loader:
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0].numpy()
            label_num += np.sum(mask_lab == 1)
            unlabel_num += np.sum(mask_lab == 0)
        label_nums.append(label_num)
        unlabel_nums.append(unlabel_num)

    print('labeled subset len:', label_nums)
    print('unlabeled subset len:', unlabel_nums)
    subset_nums = label_nums + unlabel_nums

    with open (f'{subset_len_path}/{args.dataset_name}_subset_len.txt', 'w') as f:
        for num in subset_nums:
            f.write(str(num)+'\n')
