from functools import lru_cache
import os
import glob
import json
from typing import Optional, Union

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset
from . import lookup_collator
from ..data.pkl_datasets import build_dataset

class BatchedDataDataset(FairseqDataset):
    def __init__(self, dataset, max_edge=512, cfg=None):
        super().__init__()
        self.dataset = dataset
        self.max_edge = max_edge
        self.collator = lookup_collator(cfg.dataset_source, cfg.task)

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return self.collator(
            samples,
            max_edge=self.max_edge
        )

class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        _, _, y = self.dataset[index]
        return y
    
    def __len__(self):
        return len(self.dataset)
    
    def collator(self, samples):
        return torch.stack(samples, dim=0)


class Trans2FormerDataset:
    def __init__(self,cfg = None):
        super().__init__()
        self.seed = cfg.seed
        self.cfg = cfg
        self.setup()

    def setup(self):
        (
            self.train_idx, 
            self.valid_idx, 
            self.test_idx, 
            self.dataset_train, 
            self.dataset_valid, 
            self.dataset_test 
        ) = build_dataset(self.seed, self.cfg)
    
    
class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)
    
    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)
    
    def ordered_indices(self):
        return self.sort_order
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False