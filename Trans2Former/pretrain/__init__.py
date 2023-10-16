from torch.hub import load_state_dict_from_url
import torch.distributed as dist
import torch
import os
import glob
from os import path

def load_pretrained_model(default_ckpt_dir, pretrained_model_name: str):
    if pretrained_model_name.endswith('.pt'):
        pretrain_model = pretrained_model_name.split('.')[0].split('/')[-1]
        path = os.path.join(default_ckpt_dir, pretrain_model, pretrained_model_name)
        if not os.path.exists(path):
            raise ValueError("Cannot find pretrained model name %s with path: %s.", pretrained_model_name, path)
    else:
        path = os.path.join(default_ckpt_dir, pretrained_model_name, pretrained_model_name+'_best.pt')
        if not os.path.exists(path):
            raise ValueError("Cannot find pretrained model name %s with path: %s.", pretrained_model_name+'_best.pt', path)
    # load the model
    if not dist.is_initialized():
        return torch.load(path)['model']
    else:
        pretrain_model = torch.load(path)['model']
        dist.barrier()
        return pretrain_model

