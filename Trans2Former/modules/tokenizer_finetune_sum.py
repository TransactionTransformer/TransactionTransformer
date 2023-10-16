import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .orf import gaussian_orthogonal_random_matrix_batched

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
        num_nodes,
        num_edges,
        rand_node_id,
        rand_node_id_dim,
        orf_node_id,
        orf_node_id_dim,
        type_id,
        etype_id,
        func_id,
        token_id,
        hidden_dim,
        n_layers, 
    ):
        super(FinetuneEmbedding, self).__init__()

        self.encoder_embed_dim = hidden_dim
        # TokenGT use index connection to embedding, i.e. [[u_i,...],[v_i,...]]
        # therefore the embedding is [E(u_i, v_i), ...] for each graph
        # self.edge_encoder = nn.Embedding(num_edges, hidden_dim, padding_idx=0) 
        self.edge_encoder = nn.Linear(4, hidden_dim)
        # self.node_encoder = nn.Embedding(num_nodes, hidden_dim, padding_idx=0)
        # self.node_encoder = nn.Embedding(65537, hidden_dim, padding_idx=0)
        self.node_encoder = nn.Linear(4, hidden_dim)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.seperate_token = nn.Embedding(1, hidden_dim) # indicator for seperation
        
        # Add random feature to increase topology
        self.rand_node_id = rand_node_id
        if self.rand_node_id:
            self.rand_node_id_dim = rand_node_id_dim
            self.rand_encoder = nn.Linear(2 * rand_node_id_dim, hidden_dim, bias=False)
        
        self.orf_node_id = orf_node_id
        if self.orf_node_id:
            self.orf_node_id_dim = orf_node_id_dim
            self.orf_encoder = nn.Linear(2 * orf_node_id_dim, hidden_dim, bias=False)
        # TokenGT use this to distinguish node and edge
        self.type_id = type_id # whether to use typeId
        if self.type_id:
            self.order_encoder = nn.Embedding(3, hidden_dim) # 
        # Do we need more embedding, for example
        #   Edge type embedding
        self.etype_id = etype_id
        if self.etype_id:
            self.etype_encoder = nn.Embedding(4, hidden_dim)
        #   Func embedding
        self.func_id = func_id
        if self.func_id:
            self.func_encoder = nn.Embedding(1800000, hidden_dim) #1656558
        #   Token Embedding
        self.token_id = token_id
        if self.token_id:
            self.token_encoder = nn.Embedding(319389, hidden_dim)
        
        
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]
        node_mask = torch.less(node_index, node_num) # [B, max_n]
        return node_mask
    
    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """

        Parameters:
        ----------------------
        node_id: Tensor([sum(node_num), D])
        node_mask: BoolTensor([B, max_n])
        padded_index: LongTensor([B, T, 2])

        Return:
        Tensor([B,T,2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b,max_n,2,d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index) # [B,T,2,D]
        index_embed = index_embed.view(b, max_len, 2*d)
        return index_embed
    
    def get_type_embed(self, padded_index, padding_mask):
        """
        Node Type or Edge Type

        padded_index: LongTensor([B,T,2])
        return: Tensor([B,T,D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()
        # <--Following is added after Sep 3rd, model before that times should delete
        order += 1
        order[padding_mask] = 0
        # -->
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
            if torch.any(torch.ge(embedding, 319389)):
                from fairseq.trainer import ForkedPdb; ForkedPdb().set_trace()
            if torch.any(torch.lt(embedding, 0)):
                from fairseq.trainer import ForkedPdb; ForkedPdb().set_trace()
                
            embedding = self.token_encoder(embedding)
        padded_feature[padded_edge_mask, :] += embedding
        return padded_feature
    
    def add_special_tokens(self, padded_feature, padding_mask, sep_index):
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
        sep_token_feature = self.seperate_token.weight.expand(b,1,d)
        index0 = sep_index[:,0]
        padded_feature[torch.arange(index0.size(0)),index0,:] = sep_token_feature.squeeze(1)
        
        # CLS token
        num_special_tokens = 1
        graph_token_feature = self.graph_token.weight.expand(b,1,d)
        special_token_feature = graph_token_feature
        special_token_mask = torch.zeros(b, num_special_tokens, dtype=torch.bool, device=padded_feature.device)

        padded_feature = torch.cat((special_token_feature, padded_feature), dim=1)
        padding_mask = torch.cat((special_token_mask, padding_mask), dim=1)

        return padded_feature, padding_mask
    
    @staticmethod
    def get_batch(node_features, edge_indices, edge_features, node_nums, edge_nums, perturb):
        """
        Concat node_feature and edge_feature, pad the result feature

        Parameters:
        ----------------------
        node_features: Tensor([sum(node_num), D])
        edge_indices: LongTensor([2, sum(edge_num)])
        edge_features: Tensor([sum(edge_num), D])
        node_nums: list, number of nodes in each graph
        edge_nums: list, number of edges in each graph
        perturb: Tensor([B, max(node_num), D])

        Return:
        ----------------------
        padded_index: LongTensor([B,T,2]),
        padded_feature: Tensor([B,T,D]),
        padding_mask: BoolTensor([B,T])
        """
        seq_len = [n1+e1 for n1,e1 in zip(node_nums, edge_nums)]
        b = len(seq_len)
        d = node_features.size(-1)
        max_len = max(seq_len)+2 # Two seperation token
        device = edge_indices.device

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
        sep_index = ctr_node_num+ctr_edge_num # TODO remove plus 1

        padded_index = torch.zeros(b, max_len, 2, device=device, dtype=torch.long) # [B,T,2]
        padded_index[padded_ctr_node_mask, :] = ctr_node_index.t() # Node indexing [[edge1_u, edge1_v, ...], [edge1_u, edge1_v, ...]] # |all nodes| x 2
        padded_index[padded_ctr_edge_mask, :] = edge_indices # Connected Edge Indexing [[edge1_u, ...], [edge1_v, ...]] # |all edge pair| x 2
        
        if perturb is not None:
            perturb_mask = padded_ctr_node_mask[:, :ctr_max_n] # [B, max_n]
            node_features = node_features + perturb[perturb_mask].type(node_features[0].dtype) # [sum(n_node), D]
        padded_feature = torch.zeros(b, max_len, d, device=device, dtype=node_features[0].dtype) # [B,T,D]
        padded_feature[padded_ctr_node_mask, :] = node_features
        padded_feature[padded_ctr_edge_mask, :] = edge_features
        padding_mask = torch.greater_equal(token_pos, seq_len) # [B,T] padded position should be True, while value pos should be False
        
        return padded_index, padded_feature, padding_mask, \
            padded_ctr_node_mask, padded_ctr_edge_mask, sep_index

    def forward(self, batched_data, perturb=None):

        ctr_node_data = batched_data['node_data']
        ctr_node_num = batched_data['node_num']
        ctr_edge_index = batched_data['edge_index']
        ctr_edge_data = batched_data['edge_data']
        ctr_edge_num = batched_data['edge_num']

        ctr_node_feature = self.node_encoder(ctr_node_data.float())  # [sum(n_node in each graph), D]
        ctr_edge_feature = self.edge_encoder(ctr_edge_data) # [sum(n_edge in each grpah), D]

        device = ctr_node_feature.device
        dtype = ctr_node_feature.dtype

        padded_index, padded_feature, padding_mask, ctr_node_mask, ctr_edge_mask, sep_index = self.get_batch(
            ctr_node_feature, 
            ctr_edge_index,
            ctr_edge_feature, 
            ctr_node_num,
            ctr_edge_num,
            perturb
        )
        # Orthogonalize nodes
        # ctr_node_mask = self.get_node_mask(ctr_node_num, ctr_node_feature.device) #[B, max(n_node)]
        if self.rand_node_id:
            rand_node_id = torch.rand(sum(ctr_node_num),
                                      self.rand_node_id_dim, device=device, dtype=dtype)
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(rand_node_id, ctr_node_mask, padded_index) # [B,T,2D]
            padded_feature = padded_feature + self.rand_encoder(rand_index_embed)
        
        if self.orf_node_id:
            ctr_mask = self.get_node_mask(ctr_node_num, device)
            b, max_n = padded_feature.shape[0], ctr_mask.shape[1]
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]

            if self.orf_node_id_dim > max_n:
                orf_node_id = F.pad(orf, (0, self.orf_node_id_dim - max_n), value=float('0'))  # [sum(n_node), D]
            else:
                orf_node_id = orf[..., :self.orf_node_id_dim]  # [sum(n_node), D]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            ctr_orf_node_id = orf_node_id[ctr_mask, :]
            orf_index_embed = self.get_index_embed(ctr_orf_node_id, ctr_mask, padded_index)  # [B, T, 2D]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed) # TODO not implemented yet

        # Add Customize Embedding
        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index, padding_mask)
        if self.etype_id:
            ctr_etype_embedding = batched_data['etype_embedding']
            padded_feature = self.add_custom_embed(padded_feature, ctr_etype_embedding, ctr_edge_mask, type='etype')
        if self.func_id:
            ctr_func_embedding = batched_data['func_embedding']
            padded_feature = self.add_custom_embed(padded_feature, ctr_func_embedding, ctr_edge_mask, type='func')
        if self.token_id:
            ctr_token_embedding = batched_data['token_embedding']
            padded_feature = self.add_custom_embed(padded_feature, ctr_token_embedding, ctr_edge_mask, type='token')

        padded_feature, padding_mask = self.add_special_tokens(padded_feature, padding_mask, sep_index)
        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float('0'))

        return padded_feature, padding_mask, padded_index  # [B, 2+T, D], [B, 2+T], [B, T, 2]
