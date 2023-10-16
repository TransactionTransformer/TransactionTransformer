import torch
import datetime
import random

from collections import defaultdict
import pickle as pkl

tokenIdx2maxVal = pkl.load(open('./tokenIdx2maxVal.pkl','rb'))

def compact_redundent(etype_txns):
    try:
        lasttxnRep = (','.join(map(str, etype_txns[0][:7])), etype_txns[0])
    except IndexError:
        import pdb;pdb.set_trace()
    result = []
    duplicate_times = 1
    for txn in etype_txns[1:]:
        txnRep = ','.join(map(str, [txn[:7]]))
        if lasttxnRep[0] == txnRep:
            duplicate_times += 1
        else:
            result.append([lasttxnRep[1], duplicate_times])
            lasttxnRep = (txnRep, txn)
            duplicate_times = 1
    result.append([lasttxnRep[1], duplicate_times])
    return result

def format_data_helper(txns, txnIdx):
    """
    Helper function for format_data: iterate every transation
    and compact redundent subtransactions. 
    """
    feats = []
    froms, tos = [], []
    funcs, tokens, etypes = [], [], []
    for etypeidx, etype in enumerate(['external', 'internal', 'erc20', 'erc721']):
        if etype in txns and len(txns[etype]) > 0:
            compact_txns = compact_redundent(txns[etype])
            for txn, duplicate_times in compact_txns:
                froms.append(txn[0])
                tos.append(txn[1])
                funcs.append(txn[2])
                tokens.append(txn[3])
                etypes.append(etypeidx)
                # [value, fromIsContract, toIsContract, duplicate_times]
                if txn[4] == 4 and etypeidx == 3:   # NFT
                    value = 1
                elif txn[3] in tokenIdx2maxVal:     # if token
                    value = txn[4] / tokenIdx2maxVal[int(txn[3])]
                else:
                    value = txn[4]/1e18
                    if value > torch.iinfo(torch.long).max:
                        value = torch.iinfo(torch.long).max
                feats.append([value, txn[5], txn[6], duplicate_times])
    # Edge Index
    edge_index = list(zip(froms, tos))
    # Func
    func_embedding = funcs
    # Token
    token_embedding = tokens
    # Etype
    etypes_embedding = etypes
    # bn ts
    timeobj = datetime.datetime.fromtimestamp(txns['timestamp'])
    bnts_embedding = [[txns['blockNumber'], txnIdx, timeobj.year, timeobj.month, timeobj.weekday(), timeobj.hour]] * len(feats)

    return feats, edge_index, func_embedding, token_embedding, etypes_embedding, bnts_embedding

def format_data(data, max_edge, isNCP=False):
    """
    Flatten all txns in data, return structured data with embedding
    
    Parameter:
    -----------------------------
    data: dict of txns: dict, one entire transaction with 
        {txnIdx: {
            'blockNumber': int, 'timestamp': int, 'txnHash': str,
            'external': [...], 'internal': [...], 'erc20': [...], 'erc721': [...],
        }}
    max_edge: int, maximum input edges
    isNCP: NCP data would be like twohop data, 

    Return:
    -----------------------------
    node_data: list node features [in_degree, out_degree, in_txns, out_txns, ...] # correspond to relative idx # |sum(n_nodes)| x 4
    edge_data: list of [value, fromIsContract, toIsContract, etype] # |E| x 6
    edge_index: list of [fromIdx, toIdx] # relative idx, not absolute idx # |E| x 2
    func_embedding: list of [funcIdx]
    token_embedding: list of [tokenIdx]
    bnts_embedding: list of [blockNumber, txnIdx, month, day, hour]
    hop_embedding: list of [hop]
    """
    edge_data, edge_indices, func_embedding, token_embedding, etype_embedding, bnts_embedding = [], [], [], [], [], []
    # ONE-HOP
    for txnIdx in sorted(list(data['onehop'].keys())):
        temp = format_data_helper(data['onehop'][txnIdx], txnIdx)
        edge_data += temp[0]
        edge_indices += temp[1]
        func_embedding += temp[2]
        token_embedding += temp[3]
        etype_embedding += temp[4]
        bnts_embedding += temp[5]
    onehop_length = len(edge_data)

    # TWO-HOP
    if 'twohop' in data and not isNCP:
        for neighbor in data['twohop']:
            for txnIdx in sorted(list(data['twohop'][neighbor]['data'].keys())):
                temp = format_data_helper(data['twohop'][neighbor]['data'][txnIdx], txnIdx)
                edge_data += temp[0]
                edge_indices += temp[1]
                func_embedding += temp[2]
                token_embedding += temp[3]
                etype_embedding += temp[4]
                bnts_embedding += temp[5]
    twohop_length = len(edge_data) - onehop_length
    hop_embedding = [1] * onehop_length + [2] * twohop_length
    # Convert to tensor
    if torch.tensor(edge_data).isinf().any():
        import pdb;pdb.set_trace()
    edge_data = torch.tensor(edge_data)

    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
    func_embedding = torch.tensor(func_embedding, dtype=torch.long)
    token_embedding = torch.tensor(token_embedding, dtype=torch.long)
    etype_embedding = torch.tensor(etype_embedding, dtype=torch.long)
    bnts_embedding = torch.tensor(bnts_embedding, dtype=torch.long)
    hop_embedding = torch.tensor(hop_embedding, dtype=torch.long)

    # Truncate: current include external txns first
    # TODO: optional include only related txns first
    if edge_data.size(0) > max_edge:
        must_include_indices = torch.where(edge_data[:, -1] == 0)[0]
        if must_include_indices.size(0) >= max_edge:
            sampled_indices = must_include_indices[:max_edge]
        else: # then consider sub-transactions
            sample_size = max_edge - must_include_indices.size(0)
            not_include_indices = torch.where(edge_data[:, -1] != 0)[0]
            sampled_indices = not_include_indices[torch.randperm(len(not_include_indices))[:sample_size]]
            sampled_indices = torch.cat([sampled_indices, must_include_indices])
        sampled_indices = sampled_indices.sort().values
        # select
        edge_data = edge_data[sampled_indices]
        edge_indices = edge_indices[sampled_indices]
        func_embedding = func_embedding[sampled_indices]
        token_embedding = token_embedding[sampled_indices]
        etype_embedding = etype_embedding[sampled_indices]
        bnts_embedding = bnts_embedding[sampled_indices]
        hop_embedding = hop_embedding[sampled_indices]

    # Edge Index: Using relative index rather than definite index 
    try:
        from_iscontract = edge_data[:,1].type(torch.int8)
    except IndexError:
        from fairseq.trainer import ForkedPdb;ForkedPdb().set_trace()
    to_iscontract = edge_data[:,2].type(torch.int8)
    froms = edge_indices[:,0] * 10 + from_iscontract
    tos = edge_indices[:,1] * 10 + to_iscontract
    relative_index_dict = dict() # TODO: make center address as 0d
    for f,t in list(zip(froms.tolist(), tos.tolist())):
        if f not in relative_index_dict: relative_index_dict[f] = len(relative_index_dict)
        if t not in relative_index_dict: relative_index_dict[t] = len(relative_index_dict)
    for i in range(len(edge_indices)):
        edge_indices[i][0] = relative_index_dict[(edge_indices[i][0]*10+from_iscontract[i]).item()]
        edge_indices[i][1] = relative_index_dict[(edge_indices[i][1]*10+to_iscontract[i]).item()]
    
    # Node Data
    node_data_dict = data['node_data']
    reverse_relative_index_dict = {v: k for k,v in relative_index_dict.items()}
    node_data = []
    for i in range(len(reverse_relative_index_dict)):
        key = reverse_relative_index_dict[i]
        node_data.append(node_data_dict[key])

    node_data = torch.stack(node_data)
    hop_embedding = hop_embedding.unsqueeze(1)
    return node_data, edge_data, edge_indices, \
        func_embedding, token_embedding, bnts_embedding, etype_embedding, hop_embedding

def collator(batched, max_edge=512):
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
        center_data, ncp_data, label = items
        (center_node_data, center_edge_data, center_edge_indices, \
            cfe, cte, cbe, cee, che) = format_data(center_data, max_edge=1024)
        (ncp_node_data, ncp_edge_data, ncp_edge_indices, \
            nfe, nte, nbe, nee, nhe) = format_data(ncp_data, max_edge=16)
        
        # NODE
        node_data.append(center_node_data)
        node_data.append(ncp_node_data)
        node_num.append(len(center_node_data))
        node_num.append(len(ncp_node_data))
        # EDGE
        edge_data.append(center_edge_data)
        edge_data.append(ncp_edge_data)
        edge_indices.append(center_edge_indices)
        edge_indices.append(ncp_edge_indices)
        edge_num.append(len(center_edge_indices))
        edge_num.append(len(ncp_edge_indices))
        # Embedding
        func_embedding.append(cfe)
        func_embedding.append(nfe)
        token_embedding.append(cte)
        token_embedding.append(nte)
        bnts_embedding.append(cbe)
        bnts_embedding.append(nbe)
        hop_embedding.append(che)
        hop_embedding.append(nhe)
        etype_embedding.append(cee)
        etype_embedding.append(nee)
        # Label
        labels.append(label)

    ctr_node_data = torch.cat(node_data[::2],)
    ctr_node_data[ctr_node_data > 65536] = 65536        
    ncp_node_data = torch.cat(node_data[1::2])
    ncp_node_data[ncp_node_data > 65536] = 65536
    ctr_edge_data = torch.cat(edge_data[::2])
    ncp_edge_data = torch.cat(edge_data[1::2])
    ctr_edge_index = torch.cat(edge_indices[::2])
    ncp_edge_index = torch.cat(edge_indices[1::2])
    return dict(
        node_data       = (ctr_node_data, ncp_node_data),
        node_num        = (torch.tensor(node_num[::2]), torch.tensor(node_num[1::2])),
        edge_index      = (ctr_edge_index, ncp_edge_index),
        edge_data       = (ctr_edge_data, ncp_edge_data),
        edge_num        = (torch.tensor(edge_num[::2]), torch.tensor(edge_num[1::2])),
        func_embedding  = (func_embedding[::2], func_embedding[1::2]),
        token_embedding = (token_embedding[::2], token_embedding[1::2]),
        bnts_embedding  = (bnts_embedding[::2], bnts_embedding[1::2]),
        hop_embedding   = (hop_embedding[::2], hop_embedding[1::2]),
        etype_embedding = (etype_embedding[::2], etype_embedding[1::2]),
        labels          = labels
    )


