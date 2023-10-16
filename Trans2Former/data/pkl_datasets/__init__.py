import os
import glob
import random
import torch
import pickle as pkl
import numpy as np
from .pretrain_dataset import Trans2FormerPretrainDataset
from .finetune_dataset import Trans2FormerFinetuneDataset

from sklearn.model_selection import train_test_split

def build_dataset(seed, cfg):
    if cfg.task == 'pretrain':
        assert os.path.exists(cfg.data_dir), 'Folder not exits'
        nodes = open(os.path.join(cfg.data_dir, 'total_nodes_new.txt'),'r').readline()
        nodes = nodes.split(',')
        mapping_fp = os.path.join(cfg.data_dir, 'NCP_mapping_total_new.pkl')
        num_data = len(nodes)

        # Generate Train/Valid/Test indices
        train_valid_idx, test_idx = train_test_split(
            np.arange(num_data),
            test_size=num_data//100,
            random_state=seed,
        )
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=num_data//50, random_state=seed
        )
        train_files = [nodes[idx] for idx in train_idx]
        valid_files = [nodes[idx] for idx in valid_idx]
        test_files = [nodes[idx] for idx in test_idx]
        
        dataset_cls = Trans2FormerPretrainDataset

        dataset_train = dataset_cls(train_files, mapping_fp, seed, cfg)
        dataset_valid = dataset_cls(valid_files, mapping_fp, seed, cfg)
        dataset_test = dataset_cls(test_files, mapping_fp, seed, cfg)

    elif cfg.task == 'finetune':
        nodes = glob.glob(os.path.join(cfg.data_dir, '*/*.pkl'))
        label_file = os.path.join(cfg.data_dir, 'label.pkl')
        num_data = len(nodes)

        # Generate Train/Valid/Test indices
        train_valid_idx, test_idx = train_test_split(
            np.arange(num_data),
            test_size=num_data//10,
            random_state=seed,
        )
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=num_data//5, random_state=seed
        )
        train_graphs = [nodes[idx] for idx in train_idx]
        valid_graphs = [nodes[idx] for idx in valid_idx]
        test_graphs = [nodes[idx] for idx in test_idx]

        dataset_cls = Trans2FormerFinetuneDataset
        dataset_train = dataset_cls(train_graphs, label_file, seed, cfg)
        dataset_valid = dataset_cls(valid_graphs, label_file, seed, cfg)
        dataset_test = dataset_cls(test_graphs, label_file, seed, cfg)

    return train_idx, valid_idx, test_idx, dataset_train, dataset_valid, dataset_test

