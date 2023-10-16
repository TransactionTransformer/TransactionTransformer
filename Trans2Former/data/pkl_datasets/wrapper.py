import os
import json
import torch
import numpy as np
import torch_geometric
from functools import lru_cache, cmp_to_key
from collections import defaultdict
import fcntl
import torch.distributed as dist
import web3
from decimal import Decimal

import time
from datetime import datetime
w3 = web3.Web3()

def convert_wei_to_ether(value_str):
    """
    Convert wei to ether in torch.float32.
    TODO lifan: Notice that precision may be compromised.
    """
    if isinstance(value_str, str): value = Decimal(value_str)
    else: value = value_str
    value = torch.tensor(w3.fromWei(value, 'ether'), dtype=torch.float32)
    if torch.isinf(value):
        return torch.tensor(torch.iinfo(torch.int32).max, dtype=torch.float32)
    return value

def convert_token_value(value_str, token_range):
    if isinstance(value_str, str): value = int(value_str)
    else: value = value_str
    value = torch.tensor(value/token_range) if token_range != 0 else torch.tensor(value)

    return value

def convert_etype(etype):
    if etype == 'external': return 1
    elif etype == 'internal': return 2
    elif etype == 'erc20': return 3
    elif etype == 'erc721': return 4
    else:
        raise ValueError(f'{etype} is not included as edge types')

def convert_timestamp(timestamp):
    timestamp = int(timestamp)
    dtobj = datetime.fromtimestamp(timestamp)
    return dtobj.year, dtobj.month, dtobj.day, dtobj.hour

@torch.jit.script
def convert_to_single_emb(x, offset: int=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def converter(txn, token_range:int=1):
    if isinstance(txn, dict):
        y, m, d, h = convert_timestamp(txn['timestamp'])
        if txn['type'] == 1 or txn['type'] == 2:
            value = convert_wei_to_ether(txn['value'])
        elif txn['type'] == 3:
            value = convert_token_value(txn['value'], token_range)
        elif txn['type'] == 4:
            value = torch.tensor(1)
        data = torch.tensor([
            torch.tensor(int(txn['blockNumber'])),
            torch.tensor(y),
            torch.tensor(m),
            torch.tensor(d), # TODO lifan: embed_dim must be divisible by num_heads
            torch.tensor(h),
            value,
            convert_wei_to_ether(txn['gasLimit']),
            convert_wei_to_ether(txn['gasUsed']),
            convert_wei_to_ether(txn['gasPrice']),
            torch.tensor(txn['type']),
        ])
        return data
    elif isinstance(txn, list) or isinstance(txn, tuple):
        y, m, d, h = convert_timestamp(txn[1])
        data = torch.tensor([
            torch.tensor(int(txn[0])),
            torch.tensor(y),
            torch.tensor(m),
            torch.tensor(d), # TODO lifan: embed_dim must be divisible by num_heads
            torch.tensor(h),
            convert_wei_to_ether(txn[2]) if len(txn) == 10 else torch.tensor(0),
            convert_wei_to_ether(txn[3]) if len(txn) == 10 else torch.tensor(0),
            convert_wei_to_ether(txn[4]) if len(txn) == 10 else torch.tensor(0),
            convert_wei_to_ether(txn[5]) if len(txn) == 10 else torch.tensor(0),
            torch.tensor(txn[-1]),
        ])
        return data
    else:
        raise KeyError
    
def cmp(x1, x2):
    """Sort by blockNumber(pos==0), then by linenumber (pos==-2)"""
    x1s = x1.split(',') 
    x2s = x2.split(',')
    if int(x1s[0]) < int(x2s[0]): return -1
    elif int(x1s[0]) > int(x2s[0]): return 1
    else:
        return -1 if int(x1s[-2]) < int(x2s[-2]) else 1
            