import math

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from torch.utils.data import Dataset


def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO: Create two DataLoader instances, dl_train and dl_valid.
    # They should together represent a train/validation split of the given
    # dataset. Make sure that:
    # 1. Validation set size is validation_ratio * total number of samples.
    # 2. No sample is in both datasets. You can select samples at random
    #    from the dataset.

    # ====== YOUR CODE: ======
    test_indices, train_indices = get_indices(dataset, validation_ratio)
    dl_train, dl_valid = get_dl(dataset, train_indices, test_indices, batch_size, num_workers)
    # ========================

    return dl_train, dl_valid

# ====== MORE OF YOUR CODE: ======
def get_indices(dataset, validation_ratio, part=0, random=True, seed=41):
    indices = torch.tensor(list(range(len(dataset))))
    if random:
        torch.manual_seed(seed)
        indices = torch.randperm(len(dataset))
    seperator = int(np.floor(validation_ratio * len(dataset)))
    assert part * seperator <= len(dataset)
    assert part >= 0
    if part > 0:
        train_pt1 = indices[:seperator * part]
        train_pt2 = indices[seperator * (part + 1):]
        train_indices = torch.cat((train_pt1, train_pt2))
        test_indices = indices[seperator * part:seperator * (part + 1)]
        return test_indices, train_indices
    return indices[:seperator], indices[seperator:]


def get_dl(dataset: Dataset, train_indices, test_indices, batch_size=100, num_workers=2):
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_indices), num_workers=num_workers)
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(test_indices), num_workers=num_workers)
    return dl_train, dl_valid
# ================================