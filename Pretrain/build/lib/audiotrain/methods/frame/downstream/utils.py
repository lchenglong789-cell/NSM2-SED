from pathlib import Path
import os
import torch
import sys
import time
import datetime

from audiossl import datasets


def load_dataset(dataset_name, data_path, fold, transform):
    if dataset_name == "audioset_b":
        dataset_train = datasets.LMDBDataset(data_path, split="train", subset=3000000, transform=transform)
        dataset_val = datasets.LMDBDataset(data_path, split="valid", transform=transform)
        dataset_test = datasets.LMDBDataset(data_path, split="eval", transform=transform)
    elif dataset_name == "audioset":
        dataset_train = datasets.LMDBDataset(data_path, split="train", subset=3000000, transform=transform)
        dataset_val = datasets.LMDBDataset(data_path, split="valid", transform=transform)
        dataset_test = datasets.LMDBDataset(data_path, split="eval", transform=transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_val, dataset_test


MULTI_LABEL = {
    "audioset_b": True,
    "audioset": True
}

NUM_LABELS = {
    "audioset_b": 527,
    "audioset": 527
}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


import numpy as np
from sklearn import metrics


class Metric:
    preds = []
    targets = []

    def __init__(self, mode="ACC"):
        self.mode = mode
        self.clear()

    def update(self, pred, target):
        self.preds.append(pred)
        self.targets.append(target)

    def clear(self):
        self.preds = []
        self.targets = []

    def compute(self):
        torch.distributed.all_reduce(self.preds)
        torch.distributed.all_reduce(self.targets)
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        if self.mode == "mAP":
            mAPs = []
            for i in range(preds.shape[-1]):
                mAPs.append(metrics.average_precision_score(targets[:, i], preds[:, i], average=None))
            x = x[~numpy.isnan(x)]
            mAP = np.mean(mAPs)
            return mAP
        else:
            acc = metrics.top_k_accuracy_score(targets, preds, k=1,
                                               labels=np.linspace(0, preds.shape[-1] - 1, preds.shape[-1]))
            return acc


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        exit()
