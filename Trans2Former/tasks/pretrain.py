import logging

import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
import importlib

import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    Trans2FormerDataset,
    EpochShuffleDataset,
)

import json
import torch
import sys
import os

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig(FairseqDataclass):
    dataset_name: str = field(
        default='ethereum',
        metadata={"help": "name of the dataset"},
    )

    max_edges: int = field(
        default=256,
        metadata={"help": "max edges per graph"},
    )

    address_dir: str = field(
        default="",
        metadata={"help": "Path to Address Index"},
    )

    data_dir: str = field(
        default="",
        metadata={"help": "Path to Dataset"},
    )

    dataset_source: str = field(
        default="pkl",
        metadata={"help": "source of dataset, can be: pkl"},
    )

    ckpt_dir: str = field(
        default=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'examples', 'ckpts'),
        metadata = {"help": "Path to model checkpoint save dir"}
    )

    pretrained_model_name: str = field(
        default="None",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    train_epoch_shuffle: bool = field(
        default=False,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )
    
    task: str = field(
        default="",
        metadata={"help": "name of the task"}
    )
    seed: int = II("common.seed")

    pretrain_task: str = field(
        default="NCP",
        metadata={"help": "name of pretrain task"},
    )

    # NCP params
    fake_edge_prob: float = field(
        default=0.5,
        metadata={"help": "probability of wether pair node has connections with center node",}
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regresion targets"},
    )


@register_task("pretrain", dataclass=PretrainConfig)
class PretrainTask(FairseqTask):
    """
    Pretrain task for trans2former
    """

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        super().__init__(cfg, **kwargs)
        assert cfg.data_dir != "", "Dataset path should be provided"
        self.dm = Trans2FormerDataset(cfg=cfg)
        
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)
    
    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_valid
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(
            batched_data,
            max_edge=self.max_edges(),
            cfg=self.cfg
        )

        data_sizes = np.array([self.max_edges()] * len(batched_data))

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset, num_samples=len(dataset), seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #sample: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]
    
    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_edges
        
        model = models.build_model(cfg, self)

        with open(os.path.join(os.path.dirname(cfg.save_dir), 'parameters.json'),'w') as f:
            f.write(json.dumps(vars(cfg), indent=4))
        return model
    
    def max_edges(self):
        return self.cfg.max_edges
    
    def max_nodes(self):
        return self.cfg.max_edges
    
    @property
    def source_dictionary(self):
        return None
    
    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None
    
