from matplotlib.pyplot import get
from data.data_utils import MergedDataset
from data.herbarium_19 import get_herbarium_datasets
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets, get_cifar_50_datasets,\
    get_cifar_90_10_datasets, get_cifar_80_20_datasets, get_cifar_70_30_datasets, get_cifar_60_40_datasets,\
get_cifar_50_50_datasets, get_cifar_40_60_datasets, get_cifar_30_70_datasets, get_cifar_20_80_datasets,\
get_cifar_10_90_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pets import get_pets_datasets
from data.flower import get_flower_datasets
from data.food import get_food_datasets
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.cifar import subsample_classes as subsample_dataset_cifar
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
from data.pets import subsample_classes as subsample_dataset_pets
from data.flower import subsample_classes as subsample_dataset_flower
from data.food import subsample_classes as subsample_dataset_food

from copy import deepcopy
import pickle
import os

from config import osr_split_dir

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'cifar50': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'pets': subsample_dataset_pets,
    'flower': subsample_dataset_flower,
    'food': subsample_dataset_food
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar50': get_cifar_50_datasets,
    'cifar90_10': get_cifar_90_10_datasets,
    'cifar80_20': get_cifar_80_20_datasets,
    'cifar70_30': get_cifar_70_30_datasets,
    'cifar60_40': get_cifar_60_40_datasets,
    'cifar50_50': get_cifar_50_50_datasets,
    'cifar40_60': get_cifar_40_60_datasets,
    'cifar30_70': get_cifar_30_70_datasets,
    'cifar20_80': get_cifar_20_80_datasets,
    'cifar10_90': get_cifar_10_90_datasets,


    'imagenet_100': get_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'pets': get_pets_datasets,
    'flower': get_flower_datasets,
    'food': get_food_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    # ['train_labelled', 'train_unlabelled', 'val', 'test']
    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = deepcopy(datasets['test'])
    test_dataset.transform = test_transform
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    unlabelled_train_examples_train = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_train.transform = train_transform

    labelled_train_examples = deepcopy(datasets['train_labelled'])
    labelled_train_examples.transform = test_transform

    labelled_train_examples_attribute = deepcopy(datasets['train_labelled'])
    labelled_train_examples_attribute.transform = train_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar50':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)



    elif args.dataset_name == 'cifar90_10':

        args.image_size = 32
        args.train_classes = range(90)
        args.unlabeled_classes = range(90, 100)

    elif args.dataset_name == 'cifar80_20':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar70_30':

        args.image_size = 32
        args.train_classes = range(70)
        args.unlabeled_classes = range(70, 100)

    elif args.dataset_name == 'cifar60_40':

        args.image_size = 32
        args.train_classes = range(60)
        args.unlabeled_classes = range(60, 100)

    elif args.dataset_name == 'cifar50_50':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cifar40_60':

        args.image_size = 32
        args.train_classes = range(40)
        args.unlabeled_classes = range(40, 100)

    elif args.dataset_name == 'cifar30_70':

        args.image_size = 32
        args.train_classes = range(30)
        args.unlabeled_classes = range(30, 100)

    elif args.dataset_name == 'cifar20_80':

        args.image_size = 32
        args.train_classes = range(20)
        args.unlabeled_classes = range(20, 100)
    elif args.dataset_name == 'cifar10_90':

        args.image_size = 32
        args.train_classes = range(10)
        args.unlabeled_classes = range(10, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']


    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)


    elif args.dataset_name == 'pets':

        args.image_size = 224
        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)
    
    elif args.dataset_name == 'flower':

        args.image_size = 224
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 102)

    elif args.dataset_name == 'food':

        args.image_size = 224
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 101)

    else:

        raise NotImplementedError

    return args