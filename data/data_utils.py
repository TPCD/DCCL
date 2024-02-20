import numpy as np
import torch
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices
import copy
class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset=None):
        _labelled_dataset = copy.deepcopy(labelled_dataset)
        _unlabelled_dataset = copy.deepcopy(unlabelled_dataset)

        self.labelled_dataset = _labelled_dataset
        self.unlabelled_dataset = _unlabelled_dataset

        self.target_transform = None
        if unlabelled_dataset is not None:
            if hasattr(labelled_dataset, 'data'):
                if isinstance(labelled_dataset.data, list):
                    self.data = _labelled_dataset.data + _unlabelled_dataset.data
                    if hasattr(_labelled_dataset, 'target'):
                        self.target = _labelled_dataset.target + _unlabelled_dataset.target
                    if hasattr(_labelled_dataset, 'uq_idxs'):
                        self.uq_idxs = _labelled_dataset.uq_idxs.tolist() + _unlabelled_dataset.uq_idxs.tolist()
                elif _labelled_dataset.data.shape[1] == _unlabelled_dataset.data.shape[1]:
                    self.data = np.concatenate((_labelled_dataset.data, _unlabelled_dataset.data), axis=0)
                else:
                    assert False, f'size not match: {_labelled_dataset.data.shape[1]} and {_unlabelled_dataset.data.shape[1]}'
            elif hasattr(labelled_dataset, 'samples'):
                 self.data = _labelled_dataset.samples + _unlabelled_dataset.samples
                 if hasattr(_labelled_dataset, 'target'):
                     self.target = _labelled_dataset.target + _unlabelled_dataset.target
                 if hasattr(_labelled_dataset, 'uq_idxs'):
                     self.uq_idxs = _labelled_dataset.uq_idxs.tolist() + _unlabelled_dataset.uq_idxs.tolist()
            else:
                assert False, f'Unsuport {labelled_dataset}'

    def __getitem__(self, item):
        if self.unlabelled_dataset is None:
            _tuple = self.labelled_dataset[item]
            if len(_tuple) > 3:
                img, label, uq_idx, attr = _tuple
                labeled_or_not = 1
                return img, label, uq_idx, np.array([labeled_or_not]), attr
            else:
                img, label, uq_idx = _tuple
                labeled_or_not = 1
                return img, label, uq_idx, np.array([labeled_or_not]), np.array([0])
        else:
            if item < len(self.labelled_dataset):
                _tuple = self.labelled_dataset[item]
                if len(_tuple) > 3:
                    img, label, uq_idx, attr = _tuple
                    labeled_or_not = 1
                    return img, label, uq_idx, np.array([labeled_or_not]), attr
                else:
                    img, label, uq_idx = _tuple
                    labeled_or_not = 1
                    return img, label, uq_idx, np.array([labeled_or_not]), np.array([0])
            else:
                _tuple = self.unlabelled_dataset[item - len(self.labelled_dataset)]
                if len(_tuple) > 3:
                    img, label, uq_idx, attr = _tuple
                    labeled_or_not = 0
                    return img, label, uq_idx, np.array([labeled_or_not]), attr
                else:
                    img, label, uq_idx = _tuple
                    labeled_or_not = 0
                    return img, label, uq_idx, np.array([labeled_or_not]), np.array([0])

    def __len__(self):
        if self.unlabelled_dataset is None:
            return len(self.labelled_dataset)
        else:

            return len(self.unlabelled_dataset) + len(self.labelled_dataset)

