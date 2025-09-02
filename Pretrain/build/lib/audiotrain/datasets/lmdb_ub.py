import lmdb
import pyarrow as pa

import os
import torch
import random
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

random.seed(1234)


class MultiLMDBDataset(Dataset):
    def __init__(self, db_path, split, subset=None, transform=None, target_transform=None, return_key=False):
        self.db_path = db_path
        self.return_key = return_key
        self.split = split
        self.subset = subset

        self.envs = []
        self.key_maps = []
        self.total_length = 0

        if split == "train":
            lmdb_files = sorted([f for f in os.listdir(db_path)
                                 if f.startswith("train_") and f.endswith(".lmdb")])
        elif split in ["valid", "eval"]:
            lmdb_files = [f"{split}.lmdb"]
        else:
            raise ValueError(f"Invalid split: {split}")

        for lmdb_file in lmdb_files:
            lmdb_path = os.path.join(db_path, lmdb_file)
            env = lmdb.open(lmdb_path,
                            subdir=False,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)

            with env.begin(write=False) as txn:
                length = pa.deserialize(txn.get(b'__len__'))
                keys = pa.deserialize(txn.get(b'__keys__'))

                self.envs.append({
                    'env': env,
                    'txn': env.begin(write=False),
                    'keys': keys,
                    'length': length
                })
                self.total_length += length
                self.key_maps.extend([(len(self.envs) - 1, k) for k in keys])

        if subset is not None and subset < self.total_length:
            self.total_length = subset
            random.shuffle(self.key_maps)
            self.key_maps = self.key_maps[:subset]
            self.total_length = subset

        self.transform = transform
        self.target_transform = target_transform
        self.sr = 16000

    def __getitem__(self, index):
        env_idx, key = self.key_maps[index]
        env_info = self.envs[env_idx]

        byteflow = env_info['txn'].get(key)
        unpacked = pa.deserialize(byteflow)

        waveform = torch.from_numpy(unpacked[0]).squeeze(0)
        label = torch.from_numpy(unpacked[1]).squeeze(0)

        if self.transform:
            transformed = self.transform(waveform)
            if self.target_transform:
                transformed = list(transformed)
                transformed[0], label = self.target_transform(transformed[0], label)
                transformed = tuple(transformed)

            if self.return_key:
                return transformed, label, key
            else:
                return transformed, label
        else:
            if self.return_key:
                return waveform, label, key
            else:
                return waveform, label

    def __len__(self):
        return self.total_length

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.db_path}, split={self.split}, size={len(self)})"

    def close(self):
        for env_info in self.envs:
            env_info['env'].close()
