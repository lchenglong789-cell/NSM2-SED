from pathlib import Path
import os

from .registry import register_dataset, list_all_datasets, get_dataset

from .lmdb import LMDBDataset
# from .lmdb_ub import MultiLMDBDataset
from .dcase import DCASEDataset
from .as_strong import ASStrongDataset


@register_dataset("audioset_b", multi_label=True, num_labels=527, num_folds=1)
def create_audioset_b(data_path, split, transform, target_transform, return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path, split=split, transform=transform, target_transform=target_transform,
                       return_key=return_key)


@register_dataset("audioset", multi_label=True, num_labels=527, num_folds=1)
def create_audioset(data_path, split, transform, target_transform, return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path, split=split, transform=transform, target_transform=target_transform,
                       return_key=return_key)


# [DCASE MARK] add a register for dcase dataset
@register_dataset("dcase", multi_label=True, num_labels=10, num_folds=1)
def create_dcase(config_path, split, transform=None, target_transform=None, unsup=False, return_key=False):
    assert split in ["train", "valid", "test"], "Dataset type: {} is not supported.".format(split)
    return DCASEDataset(config_path, split, transform=transform, target_transform=None, unsup=unsup)


@register_dataset("as_strong", multi_label=True, num_labels=407, num_folds=1)
def create_dcase(as_strong_conf, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], "Dataset type: {} is not supported.".format(split)
    return ASStrongDataset(as_strong_conf, split, transform=transform, target_transform=None)


__all__ = ['LMDBDataset']
