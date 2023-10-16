import os
import glob
import json
import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from functools import lru_cache

class Trans2FormerFinetuneDataset(Dataset):
    """
    Pretrain Dataset for Trans2Former loading the transaction data from json.
    """
    def __init__(self, pkl_files, label_file, seed, cfg) -> None:
        super().__init__()
        self.seed = seed
        self.pkl_files = pkl_files
        self.label = pkl.load(open(label_file,'rb'))
        self.cfg = cfg

    def __len__(self):
        return len(self.pkl_files)

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        fp = self.pkl_files(index)
        node_address = os.path.basename(fp)[:-4]
        label = self.label[node_address]
        data = pkl.load(open(fp, 'rb'))
        return data, node_address, label