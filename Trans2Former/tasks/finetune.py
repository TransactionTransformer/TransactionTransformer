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
from ..data.dataset import(
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
class FinetuneConfig(FairseqDataclass):
    dataset_name: str = field(
        default='phishing',
        metadata={"help": "name of the dataset"}
    )

    max_edges: int = field(
        default=256,
        metadata={"help": "max edges per graph"}
    )

    address_file: str = field(
        default="",
        metadata={"help": "Path to Address Index"},
    )

    data_dir: str = field(
        default="",
        metadata={"help": "Path to Dataset"},
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
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regresion targets"},
    )
    task: str = field(
        default="",
        metadata={"help": "name of the task"}
    )
    seed: int = II("common.seed")

@register_task("finetune", dataclass=FinetuneConfig)
class FinetuneTask(FairseqTask):
    """
    Finetune task for edgeformer
    """

    def __init__(self, cfg: FairseqDataclass, **kwargs):
        super().__init__(cfg, **kwargs)
        assert cfg.data_dir != "", "Dataset path should be provided"

        self.dm = Trans2FormerDataset(cfg=cfg)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)
    
    def load_dataset(self, split: str, combine: bool = False, task_cfg: FairseqDataclass = None, **kwargs):
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
            max_edge=self.max_edges()
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
    
