import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.onnx.operators
from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

def positional_embedding(hidden_dim):
    return SinusoidalPositionalEmbedding(hidden_dim)

class BaseAbsoluteEmbedding(nn.Module):
    """This module produces sinusodial positional embedding of any length.
    
    """
    def __init__(self, num_embedding, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None
    
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.
        Copied from fairseq/modules/sinusoidal_positional_embedding.py
        
        """
        import pdb;pdb.set_trace()
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def forward(self, input):
        pass


class AbsoluteEmbedding(BaseAbsoluteEmbedding):
    """This module produces sinusodial positional embedding of any length.
    The position is substitued by block number, which is different from original 
    paper's defintion. 
    
    """
    def __init__(self, num_embedding, embedding_dim, padding_idx=None):
        super().__init__(embedding_dim, padding_idx, num_embedding)
        self.weights = BaseAbsoluteEmbedding.get_embedding(
            num_embedding, embedding_dim, padding_idx
        )

    def forward(self, input):
        """Input is expected to be of size [bsz, seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = BaseAbsoluteEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        return self.weights[positional_embedding].detach()
    
# class BlockNumberRelativeEmbedding(nn.Module):
#     """This module produces sinusodial positional embedding of any length.
#     The position is substitued by block number, which is different from original 
#     paper's defintion. 
    
#     """


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)
        self.register_buffer('position_ids', torch.arange(embedding_dim))

    def forward(self, x: torch.Tensor):
        """
        return (b l d) / (b l h d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]
        
        elif x.dim() == 4:
            h = x.size(1)
            shape = x.shape
            x = x.view(shape[0], shape[1], -1)
            x = x + self.embeddings(position_ids)[None, :, :]
            x = x.view(shape).permute(0,2,1,3)
            return x

# 可以考虑将编码相乘替代相加