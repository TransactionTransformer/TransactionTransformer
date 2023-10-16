from .collator_token_pretrain import *

def collator_finetune(batched, max_edge=512):
    node_data = []
    edge_data = []
    edge_indices = []
    func_embedding, token_embedding, bnts_embedding, etype_embedding, hop_embedding = [], [], [], [], []
    node_num, edge_num = [], []
    labels = []
    paddings = []
    for idx, items in enumerate(batched):
        # node_data, edge_data, edge_indices, \
        # func_embedding, token_embedding, bnts_embedding, hop_embedding
        center_data, node_address, label = items
        (center_node_data, center_edge_data, center_edge_indices, \
            cfe, cte, cbe, cee, che) = format_data(center_data, max_edge)
        
        # NODE
        node_data.append(center_node_data)
        node_num.append(len(center_node_data))
        # EDGE
        edge_data.append(center_edge_data)
        edge_indices.append(center_edge_indices)
        edge_num.append(len(center_edge_indices))
        # Embedding
        func_embedding.append(cfe)
        token_embedding.append(cte)
        bnts_embedding.append(cbe)
        hop_embedding.append(che)
        etype_embedding.append(cee)
        # Label
        labels.append(label)

    ctr_node_data = torch.tensor(node_data)
    ctr_node_data[ctr_node_data > 65536] = 65536
    ctr_edge_data = torch.cat(edge_data)
    ctr_edge_index = torch.cat(edge_indices[::2])

    return dict(
        node_data       = ctr_node_data,
        node_num        = torch.tensor(node_num),
        edge_index      = ctr_edge_index,
        edge_data       = ctr_edge_data,
        edge_num        = torch.tensor(edge_num),
        func_embedding  = func_embedding,
        token_embedding = token_embedding,
        bnts_embedding  = bnts_embedding,
        hop_embedding   = hop_embedding,
        etype_embedding = etype_embedding,
        labels          = labels
    )