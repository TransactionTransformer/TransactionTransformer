import os
import random
import json
import pickle as pkl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

class Trans2FormerPretrainDataset(Dataset):
    """
    Pretrain Dataset for EdgeFormer loading the transaction data from pkl files.
    """
    def __init__(self, pkl_files, pair_mapping_fp, seed, cfg) -> None:
        super().__init__()
        self.seed = seed
        self.pkl_files = pkl_files
        self.pair = pkl.load(open(pair_mapping_fp,'rb'))
        self.cfg = cfg
        self.dirname = cfg.data_dir
        self.other_dirname = os.path.join('/localdata_ssd/', cfg.data_dir.split('/',2)[-1])
        
        # Define task
        if 'NCP' in cfg.pretrain_task: # Neighbor Connection Prediction
            self.use_ncp = True
            prob = cfg.fake_edge_prob
            length = len(pkl_files)
            indices = torch.randperm(length)
            self.ncp_false_indices = indices[:int(length*prob)]
            
    def __len__(self):
        return len(self.pkl_files)

    def _get_random_ncp_(self, node_address):
        ncp_indices = random.sample(range(self.__len__()), 10)
        for ncp_idx in ncp_indices:
            diridx = int(self.pkl_files[ncp_idx].split('/',1)[0])
            dirpath = self.dirname if diridx > 4 else self.other_dirname 
            ncp_fp = os.path.join(dirpath, self.pkl_files[ncp_idx])
            ncp_address = os.path.basename(ncp_fp)[:-4]
            if node_address not in self.pair or \
                ncp_address not in self.pair[node_address][1]:
                return ncp_fp, False
        return ncp_fp, True
    
    def _get_neighbor_ncp_(self, node_address):
        diridx, ncp_address = self.pair[node_address]
        if isinstance(ncp_address, set) and len(ncp_address) > 0:
            ncp_address = random.sample(ncp_address, 1)[0]
        elif isinstance(ncp_address, str):
            pass
        else:
            raise NotImplementedError('{} pair cannot be handled.'.format(node_address))
        dirpath = self.dirname if diridx > 4 else self.other_dirname 
        ncp_fp = '{}/{}/NCP/{}/{}.pkl'.format(dirpath, diridx, ncp_address[2:4], ncp_address)
        if os.path.exists(ncp_fp):
            return ncp_fp, True
        else:
            return self._get_random_ncp_(node_address)

    def __getitem__(self, index):
        diridx = int(self.pkl_files[index].split('/',1)[0])
        dirpath = self.dirname if diridx > 4 else self.other_dirname 
        fp = os.path.join(dirpath, self.pkl_files[index])
        node_address = os.path.basename(self.pkl_files[index])[:-4]
        node_data = pkl.load(open(fp, 'rb'))
        if 'node_data' not in node_data:
            print(fp)
            from fairseq.trainer import ForkedPdb;ForkedPdb().set_trace()
        if len(node_data['onehop']) == 0:
            return self.__getitem__(random.randint(0, self.__len__()))
        label = False
        if node_address not in self.pair or index in self.ncp_false_indices:
            # chances exist that no paired NCP prepared, or Not NCP samples
            ncp_fp, label = self._get_random_ncp_(node_address)
        else:
            # select one neighbor
            ncp_fp, label = self._get_neighbor_ncp_(node_address)
        ncp_data = pkl.load(open(ncp_fp, 'rb'))

        if 'node_data' not in ncp_data:
            print(fp)
            from fairseq.trainer import ForkedPdb;ForkedPdb().set_trace()
        if len(ncp_data['onehop']) == 0:
            return self.__getitem__(random.randint(0, self.__len__()))
        
        return node_data, ncp_data, label
    