import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orf import gaussian_orthogonal_random_matrix_batched
from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.2/math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)


class FinetuneEmbedding(nn.Module):
    """
    Generate node and edge features along with embeddings 
    for each node and edge in the graph
    """
    def __init__(
        self,
        hidden_dim,
        n_layers,
        num_heads,
        dropout,
        task,
        pretrain_task,
        layer_norm_eps=1e-15,
        rand_node_id=False,
        rand_node_id_dim=64,
        orf_node_id=True,
        orf_node_id_dim=64,
        type_id=True,
        etype_id=True,
        func_id=False,
        token_id=True,
        attention_type='full_attention'
    ):
        super(FinetuneEmbedding, self).__init__()
        self.encoder_embed_dim = hidden_dim
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.seperate_token = nn.Embedding(1, hidden_dim) # indicator for seperation

        self.attention_type = attention_type
        self.use_attn_bias=False
        self.edge_encoder = nn.Linear(4, hidden_dim//num_heads)
        self.node_encoder = nn.Linear(4, hidden_dim//num_heads)

        # Add random feature to increase topology
        self.rand_node_id = rand_node_id
        if self.rand_node_id:
            self.rand_node_id_dim = rand_node_id_dim
            self.rand_encoder = nn.Linear(2 * rand_node_id_dim, hidden_dim, bias=False)

        self.orf_node_id = orf_node_id
        if self.orf_node_id:
            self.orf_node_id_dim = orf_node_id_dim
            self.orf_encoder = nn.Linear(2 * orf_node_id_dim, hidden_dim, bias=False)

        input_dim = hidden_dim//num_heads
        # TokenGT use this to distinguish node and edge
        self.type_id = type_id # whether to use typeId
        if self.type_id:
            input_dim += hidden_dim//num_heads
            self.order_encoder = nn.Embedding(2, hidden_dim//num_heads, padding_idx=0) # 
        # Do we need more embedding, for example
        #   Edge type embedding
        self.etype_id = etype_id
        if self.etype_id:
            input_dim += hidden_dim//num_heads
            self.etype_encoder = nn.Embedding(4, hidden_dim//num_heads)
        #   Func embedding
        self.func_id = func_id
        if self.func_id:
            self.func_encoder = nn.Embedding(1800000, hidden_dim//num_heads) #1656558
            input_dim += hidden_dim//num_heads
        #   Token Embedding
        self.token_id = token_id
        if self.token_id:
            self.token_encoder = nn.Embedding(400000, hidden_dim//num_heads) # 319389
            input_dim += hidden_dim//num_heads
        
        self.bnts = False
        if self.bnts:
            self.time_embedding = SinusoidalPositionalEmbedding(hidden_dim//num_heads, padding_idx=0)
            input_dim += hidden_dim//num_heads

        self.input_dim = input_dim
        self.input_each_dim = hidden_dim//num_heads

        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.encoding = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim//2),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(lambda module: init_params(module, n_layers=n_layers))
    
    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None] # [B, 1]
        node_mask = torch.less(node_index, node_num) # [B, max_n]
        return node_mask
        
    @staticmethod
    def get_index_embed(node_id, node_masks, padded_index):
        """

        Parameters:
        ----------------------
        node_id: Tensor([sum(node_num), D])
        node_masks: (BoolTensor([B, max_n]), BoolTensor([B, max_n]))
        padded_index: LongTensor([B, T, 2])

        Return:
        Tensor([B,T,2D])
        """
        b, max_n = node_masks.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)
        
        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)
        padded_node_id[node_masks] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b,max_n,2,d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index) # [B,T,2,D]
        index_embed = index_embed.view(b, max_len, 2*d)
        return index_embed
    
    def get_type_embed(self, padded_index):
        """
        padded_index: LongTensor([B,T,2])
        return: Tensor([B,T,D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()
        order_embed = self.order_encoder(order)
        return order_embed
    
    def add_custom_embed(self, padded_feature, embedding, padded_edge_mask, type):
        """
        Additional Embedding add to features

        Parameters:
        ---------------------------------------
        padded_feature: Tensor([B,T,D])
        embedding: B length of List(Tensor)
        padded_edge_mask: BoolTensor([B,T])
        type: str

        Return:
        --------------------------------------
        return: Tensor([B, T, D])
        """
        if type == 'etype':
            embedding = self.etype_encoder(torch.cat(embedding))
        elif type == 'func':
            embedding = torch.cat(embedding)
            embedding[torch.where(embedding==-1)[0]] = 0
            embedding = self.func_encoder(torch.cat(embedding))
        elif type == 'token':
            embedding = torch.cat(embedding)
            embedding[torch.where(embedding==-1)[0]] = 0
            embedding = self.token_encoder(embedding)
        
        padded_feature[padded_edge_mask, :] += embedding
        return padded_feature

    def add_special_tokens(self, padded_feature, padding_mask, padded_index, sep_index):
        """
        Insert special tokens SEP to feature's sep_index
        Add CLS token to features

        Parameters:
        -------------------------------
        padded_feature: Tensor([B,T,D])
        padding_mask: BoolTensor([B,T])
        sep_index: Tensor([B, 2]), one after center node, the other after ncp node

        Return:
        -------------------------------
        padded_feature: Tensor([B, 1 + T, D])
        padding_mask: BoolTensor([B, 1 + T])
        """
        b, _, d = padded_feature.size()
        # SEP token
        sep_token_feature = self.seperate_token.weight
        index0 = sep_index[:,0]
        padded_feature[torch.arange(index0.size(0)),index0,:] = sep_token_feature.squeeze(1)

        # CLS token
        num_special_tokens = 1
        graph_token_feature = self.graph_token.weight.expand(b,1,d)
        special_token_feature = graph_token_feature
        special_token_mask = torch.zeros(b, num_special_tokens, dtype=torch.bool, device=padded_feature.device) # attend is 1, ignore is 0

        # index 
        special_token_index = torch.zeros((b, num_special_tokens, padded_index.shape[2]),\
                                           dtype=padded_index.dtype, device=padded_index.device)
        padded_feature = torch.cat((special_token_feature, padded_feature), dim=1)
        padding_mask = torch.cat((special_token_mask, padding_mask), dim=1)
        padded_index = torch.cat((special_token_index, padded_index), dim=1)

        return padded_feature, padding_mask, padded_index
    
    @staticmethod
    def get_batch(edge_indices, node_nums, edge_nums):
        """
        Concat node_feature and edge_feature, pad the result feature

        Parameters:
        ----------------------
        edge_indices: LongTensor([2, sum(edge_num)])
        node_nums: list, number of nodes in each graph
        edge_nums: list, number of edges in each graph
        perturb: Tensor([B, max(node_num), D])

        Return:
        ----------------------
        padded_index: LongTensor([B,T,2]),
        padded_feature: Tensor([B,T,D]),
        padding_mask: BoolTensor([B,T])
        """
        seq_len = [n1+n2+e1+e2 for n1,n2,e1,e2 in zip(node_nums[0], node_nums[1], edge_nums[0], edge_nums[1])]
        b = len(seq_len)
        max_len = max(seq_len)+1 # Two seperation token
        device = edge_indices[0].device

        # position 
        token_pos = torch.arange(max_len, device=device).unsqueeze(0).expand(b, max_len)
        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long).unsqueeze(1)
        ctr_node_num = node_nums.type(torch.long).unsqueeze(1)
        ctr_edge_num = edge_nums.type(torch.long).unsqueeze(1)

        # unlike edge_index is provided, node_index need be crafted
        ctr_max_n = ctr_node_num.max()
        ctr_node_index = torch.arange(ctr_max_n, device=device, dtype=torch.long).unsqueeze(0).expand(b, ctr_max_n) # [B, max_n]
        ctr_node_index = ctr_node_index[None, ctr_node_index < ctr_node_num].repeat(2,1) # [2, sum(node_num)]

        padded_ctr_node_mask = torch.less(token_pos, ctr_node_num)
        padded_ctr_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, ctr_node_num),
            torch.less(token_pos, ctr_node_num+ctr_edge_num)
        )
        sep_index = ctr_node_num+ctr_edge_num
        
        padded_index = torch.zeros(b, max_len, 2, device=device, dtype=torch.long) # [B,T,2]
        padded_index[padded_ctr_node_mask, :] = ctr_node_index.t() # Node indexing [[edge1_u, edge1_v, ...], [edge1_u, edge1_v, ...]] # |all nodes| x 2
        padded_index[padded_ctr_edge_mask, :] = edge_indices # Connected Edge Indexing [[edge1_u, ...], [edge1_v, ...]] # |all edge pair| x 2
        
        padding_mask = torch.greater_equal(token_pos, seq_len) # [B,T] padded position should be True, while value pos should be False
        
        return padded_index, padding_mask,\
            padded_ctr_node_mask, padded_ctr_edge_mask, \
            sep_index

    def forward(self, batched_data, perturb=None):
        # 
        ctr_node_data = batched_data['node_data']
        ctr_node_num = batched_data['node_num']
        ctr_edge_index = batched_data['edge_index']
        ctr_edge_data = batched_data['edge_data']
        ctr_edge_num = batched_data['edge_num']
        ctr_attn_bias = batched_data['attn_bias']

        B = ctr_edge_num.shape[0]

        ctr_node_data = F.normalize(ctr_node_data.to(torch.float), dim=0)
        ctr_node_feature = self.node_encoder(ctr_node_data) # [sum(n_node in each graph), D]
        ctr_edge_feature = self.edge_encoder(ctr_edge_data.float()) # [sum(n_edge in each grpah), D]
        device = ctr_node_feature.device
        dtype = ctr_node_feature.dtype

        padded_index, padding_mask, ctr_node_mask, ctr_edge_mask, sep_index = self.get_batch(
            ctr_edge_index,
            ctr_node_num,
            ctr_edge_num,
        )
        # padded_index[padding_mask] += 1 # add 1 to avoid coliding with padding idx 0
        T = padding_mask.shape[1]
        padded_feature = torch.zeros(B, T, self.input_each_dim, device=device, dtype=dtype) # [B,T,D]
        padded_feature[ctr_node_mask, :] = ctr_node_feature
        padded_feature[ctr_edge_mask, :] = ctr_edge_feature

        # Add Customize Embedding
        if self.type_id:
            type_embedding = self.get_type_embed(padded_index)
            padded_feature = torch.cat((padded_feature, type_embedding), dim=2)
        if self.etype_id:
            ctr_etype_embedding = batched_data['etype_embedding']
            etype_embedding = torch.zeros(B, T, self.input_each_dim).to(device)
            etype_embedding[ctr_edge_mask] = self.etype_encoder(torch.cat(ctr_etype_embedding))
            padded_feature = torch.cat((padded_feature, etype_embedding), dim=2)
        if self.func_id:
            ctr_func_embedding = batched_data['func_embedding']
            func_embedding = torch.zeros(B, T, self.input_each_dim).to(device)
            func_embedding[ctr_edge_mask] = self.func_encoder(torch.cat(ctr_func_embedding))
            padded_feature = torch.cat((padded_feature, func_embedding), dim=2)
        if self.token_id:
            ctr_token_embedding = batched_data['token_embedding']
            token_embedding = torch.zeros(B, T, self.input_each_dim).to(device)
            token_embedding[ctr_edge_mask] = self.token_encoder(torch.cat(ctr_token_embedding)+1) # avoid [UNK]==-1 
            padded_feature = torch.cat((padded_feature, token_embedding), dim=2)

        # Encoding the concated feature together
        encoded_feature = self.encoding(padded_feature[~padding_mask])
        new_padded_feature = torch.zeros(B, T, self.encoder_embed_dim, device=device, dtype=dtype)
        new_padded_feature[~padding_mask] = encoded_feature

        # Global Tokens
        # Orthogonalize nodes
        if self.rand_node_id:
            rand_node_id = torch.rand(sum(ctr_node_num), \
                                      self.rand_node_id_dim, device=device, dtype=dtype)
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(rand_node_id, ctr_node_mask, padded_index) # [B,T,2D]
            orth_feature = self.rand_encoder(rand_index_embed)
            new_padded_feature += orth_feature

        if self.orf_node_id:
            ctr_mask = self.get_node_mask(ctr_node_num, device)
            b, max_n = new_padded_feature.shape[0], ctr_mask.shape[1]
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                orf_node_id = F.pad(orf, (0, self.orf_node_id_dim - max_n), value=float('0'))  # [sum(n_node), D]
            else:
                orf_node_id = orf[..., :self.orf_node_id_dim]  # [sum(n_node), D]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            ctr_orf_node_id = orf_node_id[ctr_mask, :]
            orf_index_embed = self.get_index_embed(ctr_orf_node_id, \
                                                   ctr_mask, \
                                                    padded_index)  # [B, T, 2D]
            orth_feature = self.orf_encoder(orf_index_embed)
            new_padded_feature += orth_feature # 

        # Time positional


        # Attention Bias
        if self.use_attn_bias:
            B,T,D = padded_feature.shape
            attn_bias = torch.zeros(B, T, T).to(device)
            for i, cab in enumerate(ctr_attn_bias):
                edge_mask = ctr_edge_mask[i]
                attn_bias[i, edge_mask.unsqueeze(1)*edge_mask.unsqueeze(0)] = cab.view(-1)
            # attn bias add speical token
            attn_bias = torch.cat((torch.zeros(B, T, 1).to(device), attn_bias), dim=2)
            attn_bias = torch.cat((torch.zeros(B, 1, T+1).to(device), attn_bias), dim=1)

        # Add Special Token
        new_padded_feature, padding_mask, padded_index = self.add_special_tokens(new_padded_feature, padding_mask, padded_index, sep_index)
        new_padded_feature = new_padded_feature.masked_fill(padding_mask[..., None], float('0'))
        padding_mask = torch.logical_not(padding_mask)
        

        # Pad length to block size's multiplies
        if self.attention_type == 'block_sparse':
            if self.use_attn_bias:
                new_padded_feature, padding_mask, padded_index, attn_bias, block_padding_len = \
                    self._pad_to_block_size(new_padded_feature, padding_mask, padded_index, attn_bias)
            else:
                new_padded_feature, padding_mask, padded_index, block_padding_len = self._pad_to_block_size(new_padded_feature, padding_mask, padded_index)
        else:
            block_padding_len = torch.tensor(0)

        padding_mask = torch.logical_not(padding_mask)
        padding_mask = padding_mask.type(torch.float)

        # Return Dict
        return_dict = dict(
            padded_feature=new_padded_feature,
            padding_mask=padding_mask,
            padded_index=padded_index,
            block_padding_len=block_padding_len
        )
        if self.use_attn_bias:
            return_dict.update(dict(
                attn_bias=attn_bias
            ))
        return return_dict
    
    def _pad_to_block_size(
        self,
        padded_feature, 
        padding_mask,
        padded_index,
        attn_bias=None
    ):
        """Pad the input length to fit block sparse"""
        block_size = self.block_size
        batch_size, seq_len, embed_dim = padded_feature.shape[:3]

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            # Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of 
            # `config.block_size`: {block_size}
            pad_to_feature = padded_feature.new_full(
                (batch_size, padding_len, embed_dim),
                0,
                dtype=torch.long,
            )
            padded_feature = torch.cat([padded_feature, pad_to_feature], dim=-2)
            pad_to_index = padded_index.new_full(
                (batch_size, padding_len, padded_index.shape[2]),
                0,
            )
            padded_index = torch.cat([padded_index, pad_to_index], dim=-2)
            if attn_bias is not None:
                attn_bias_ = torch.zeros(batch_size, padded_feature.shape[1], padded_feature.shape[1])
                attn_bias_[:, :seq_len, :seq_len] = attn_bias
                attn_bias = attn_bias_
        
        padding_mask = nn.functional.pad(
            padding_mask, (0, padding_len), value=False
        )

        # padded_feature = self.dropout(padded_feature)
        # padded_feature = self.LayerNorm(padded_feature)
        if attn_bias is not None:
            return padded_feature, padding_mask, padded_index, attn_bias, padding_len
        else:
            return padded_feature, padding_mask, padded_index, padding_len