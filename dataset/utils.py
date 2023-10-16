import pickle as pkl
import json
import glob
from tqdm import tqdm
import os
import torch
from decimal import Decimal
from collections import defaultdict
import fcntl    
from multiprocessing import Pool

def convert_etntxn(txn):
    fromIsContract = int(txn[6])
    toIsContract = int(txn[7])
    value = int(txn[8])
    gasLimit = int(txn[9])
    gasPrice = int(txn[10])
    gasUsed = int(txn[11])
    eip2718type = int(txn[14]) if txn[14] != 'None' else 0
    baseFeePerGas =  int(txn[15]) if txn[15] != 'None' else 0
    maxFeePerGas =  int(txn[16]) if txn[16] != 'None' else 0
    maxPriorityFeePerGas =  int(txn[17]) if txn[17] != 'None' else 0
    return [value,fromIsContract,toIsContract,gasLimit,gasPrice,gasUsed,\
            eip2718type,baseFeePerGas,maxFeePerGas,maxPriorityFeePerGas]

def convert_itntxn(txn):
    value= int(txn[5])
    callType, traceAddress = txn[0].split('_',1)
    fromIsContract= int(txn[3])
    toIsContract= int(txn[4])
    return [value,fromIsContract,toIsContract, traceAddress, callType]

def convert_e20txn(txn):
    value = int(txn[5])
    fromIsContract = int(txn[3])
    toIsContract = int(txn[4])
    return [value,fromIsContract,toIsContract]

def convert_e721txn(txn):
    tokenId = int(txn[5])
    fromIsContract= int(txn[3])
    toIsContract= int(txn[4])
    return [tokenId,fromIsContract,toIsContract]


def get_from_to_address(txn, etype):
    """
    Helper function to handle toCreate in external transaction.
    When creating a contract, `to` will be `None` and the contract
    address will be stored in `toCreate`.
    
    Parameters:
    txn: str, contains transaction info, see xblock transaction csv for detail.
    etype: int, indictor for edge type.

    Return:
    from_address, to_address: str, str
    """
    fromAddress, toAddress = '', ''
    fields = txn.split(',')
    if etype == 0:
        fromAddress, toAddress = fields[3], fields[4]
        fromIsContract, toIsContract = fields[6] == '1', fields[7] == '1'
        if fields[5] != 'None': # toCreate is not None
            toAddress = fields[5]
    else:
        fromAddress, toAddress = fields[1], fields[2]
        fromIsContract, toIsContract = fields[3] == '1', fields[4] == '1'
    if toAddress == '': import pdb;pdb.set_trace()
    return fromAddress, toAddress, fromIsContract, toIsContract

def mapping_account(txn, etype, eoa2idx, ca2idx):
    """
    Mapping hash account to index using dict.
    If the query dictionaries are updated, updated info
    will be a in returns.
    """
    def fetch_update(key, d, ntype):
        if key in d:
            return d[key]
        else:
            with open('/localdata_hdd1/lifan/datasets/xblock/preprocess/missing_address.txt','a') as f:
                f.write('{},{}\n'.format(key, ntype))
            return -1
    fromAddress, toAddress, fromIsContract, toIsContract = get_from_to_address(txn, etype)
    if fromIsContract:
        fromIdx = fetch_update(fromAddress, ca2idx, 1)
    else:
        fromIdx = fetch_update(fromAddress, eoa2idx, 0)
    if toIsContract:
        toIdx = fetch_update(toAddress, ca2idx, 1)
    else:
        toIdx = fetch_update(toAddress, eoa2idx, 0)
    return fromIdx, toIdx

def mapping_func(funcHash, func2idx):
    if funcHash in func2idx:
        return func2idx[funcHash]
    else:
        with open('/localdata_hdd1/lifan/datasets/xblock/preprocess/missing_func.txt','a') as f:
            f.write('{}\n'.format(funcHash))
        return -1

def mapping_token(tokenAddr, token2idx):
    if tokenAddr in token2idx:
        return token2idx[tokenAddr]
    else:
        with open('/localdata_hdd1/lifan/datasets/xblock/preprocess/missing_token.txt','a') as f:
            f.write('{}\n'.format(tokenAddr))
        return -1
    
def mapping_transaction(txns, etype, eoa2idx, ca2idx, token2idx, func2idx):
    """
    Mapping a trasaction which may include several internal,erc20,erc721 subtxns,
    into non-hash indices for embedding usage.
    
    """
    data = dict()
    if etype == 0: # if external
        assert len(txns) == 1, 'Duplicate external transactions!'
        txn = txns[0]
        fields = txn.split(',')
        data['blockNumber'] = int(fields[0])
        data['timestamp'] =  int(fields[1])
        data['txnHash'] = fields[2]
        fromIdx, toIdx = mapping_account(txn, 0, eoa2idx, ca2idx)
        funcIdx = mapping_func(fields[12], func2idx)
        tokenIdx = mapping_token('ETH', token2idx)
        feat = convert_etntxn(fields)
        data['external'] = [[fromIdx, toIdx, funcIdx, tokenIdx] + feat]
    elif etype == 1:
        data['internal'] = list()
        for txn in txns:
            fields = txn.split(',')
            if fields[7] != 'None': continue
            fromIdx, toIdx = mapping_account(txn, 1, eoa2idx, ca2idx)
            funcIdx = mapping_func(fields[6], func2idx)
            tokenIdx = mapping_token('ETH', token2idx)
            feat = convert_itntxn(fields)
            data['internal'].append([fromIdx, toIdx, funcIdx, tokenIdx] + feat)
    elif etype == 2:
        data['erc20'] = list()
        for txn in txns:
            fields = txn.split(',')
            fromIdx, toIdx = mapping_account(txn, 2, eoa2idx, ca2idx)
            funcIdx = mapping_func('0x', func2idx)
            tokenIdx = mapping_token(fields[0], token2idx)
            feat = convert_e20txn(fields)
            data['erc20'].append([fromIdx, toIdx, funcIdx, tokenIdx] + feat)
    elif etype == 3:
        data['erc721'] = list()
        for txn in txns:
            fields = txn.split(',')
            fromIdx, toIdx = mapping_account(txn, 3, eoa2idx, ca2idx)
            funcIdx = mapping_func('0x', func2idx)
            tokenIdx = mapping_token(fields[0], token2idx)
            feat = convert_e721txn(fields)
            data['erc721'].append([fromIdx, toIdx, funcIdx, tokenIdx] + feat)
    return data

def mapping_helper(data, eoa2idx, ca2idx, token2idx, func2idx):
    addr_data = dict()
    skip = 0    
    for txnIdx in data:
        if data[txnIdx][0][0].split(',')[13] != 'None':
            skip +=1
            continue
        addr_data[txnIdx-skip] = dict()
        for etype in range(4):
            if etype not in data[txnIdx]: continue
            mapped_txns = mapping_transaction(data[txnIdx][etype], etype, \
                                                eoa2idx,ca2idx,token2idx,func2idx)
            addr_data[txnIdx-skip].update(mapped_txns)
    return addr_data

def mapping(fp, save_dir, eoa2idx, ca2idx, token2idx, func2idx, save_temp=False):
    data = pkl.load(open(fp,'rb'))
    new_data = dict()
    for addr in tqdm(data):
        new_data[addr] = {'onehop': mapping_helper(data[addr], eoa2idx, ca2idx, token2idx, func2idx)}
    if save_temp:
        save_fp = '/localdata_hdd1/lifan/datasets/xblock/preprocess_node/' + os.path.basename(fp)
        with open(save_fp, 'wb') as f:
            pkl.dump(new_data, f)
    for addr in new_data:
        save_fp = '{}/{}/{}.pkl'.format(save_dir, addr[2:4], addr)
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        with open(save_fp, 'wb') as f:
            pkl.dump(new_data[addr], f)

def mapping_2hop(address_toextract, fp, save_dir, eoa2idx, ca2idx, token2idx, func2idx):
    print(fp)
    neighbor_data = pkl.load(open(fp, 'rb'))
    temp_twohop_data = dict()
    for neighbor in tqdm(neighbor_data):
        if neighbor in address_toextract:
            temp_twohop_data[neighbor] = dict()
            temp = mapping_helper(neighbor_data[neighbor], eoa2idx, ca2idx, token2idx, func2idx)
            center_list = address_toextract[neighbor]
            for l in center_list:
                center_addr = l[0]
                related_txnIdx = l[1:]
                temp_twohop_data[neighbor][center_addr] = {'data': temp, 'related_txnIdx': related_txnIdx}
    # revert the mapping direction
    revert = defaultdict(dict)
    for neighbor, values in temp_twohop_data.items():
        for center_addr, temp in values.items():
            revert[center_addr][neighbor] = temp
    # write to file
    for center_addr in tqdm(revert):
        cfp = '{}/{}/{}.pkl'.format(save_dir, center_addr[2:4], center_addr)
        with open(cfp, 'rb') as f:
            center_data = pkl.load(f)
        if 'twohop' not in center_addr:
            center_data['twohop'] = revert[center_addr]
        else:
            center_addr['twohop'].update(revert[center_addr])
        with open(cfp, 'wb') as f:
            pkl.dump(center_data, f)

def printPKL(fp):
    data = pkl.load(open(fp,'rb'))
    s = 'onehop:\n'
    for idx, txn in data['onehop'].items():
        s += '{}: \t{}...{}\n'.format(idx, txn['txnHash'][:10], txn['txnHash'][-8:])
    s += 'twohop:\n with {} neighbors\n'.format(len(data['twohop']))
    for addr in data['twohop']:
        s += '{}...{}: contains {} txns, related to {}\n'.format(
            addr[:6], addr[-4:],
            len(data['twohop'][addr]['data']),
            ','.join(map(str, data['twohop'][addr]['related_txnIdx']))
        )
    print(s)

def insert_node_data(fp, eoa_degrees, eoa_txns, ca_degrees, ca_txns):
    """Get node's transactions and neighbor number as its feature"""
    try:
        with open(fp, 'rb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            data = pkl.load(f)
    except EOFError:
        import pdb;pdb.set_trace()
        return
    # ONEHOP
    node_idx= set()
    for txnIdx in data['onehop']:
        txns = data['onehop'][txnIdx]
        for etype in ['external','internal','erc20','erc721']:
            if etype not in txns: continue
            for txn in txns[etype]:
                node_idx.add((txn[0], txn[5]))
                node_idx.add((txn[1], txn[6]))
    # TWOHOP
    if 'twohop' in data:
        for addr in data['twohop']:
            for txnIdx in data['twohop'][addr]['data']:
                txns = data['twohop'][addr]['data'][txnIdx]
                for etype in ['external','internal','erc20','erc721']:
                    if etype not in txns: continue
                    for txn in txns[etype]:
                        node_idx.add((txn[0], txn[5]))
                        node_idx.add((txn[1], txn[6]))
    # dict
    node_data = dict()
    for addr, is_contract in node_idx:
        if addr == -1:
            feat = [0, 0, 0, 0]
            node_data[addr*10+is_contract] = torch.tensor(feat, dtype=torch.int32)
            continue
        feat = []
        if is_contract:
            feat = torch.cat((ca_degrees[addr], ca_txns[addr])).type(torch.int32)
        else:
            feat = torch.cat((eoa_degrees[addr], eoa_txns[addr])).type(torch.int32)
        node_data[addr*10+is_contract] = feat
    data['node_data'] = node_data
    save_fp = fp
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    with open(save_fp, 'wb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        pkl.dump(data,f)

def insert(chunk_files, eoa_degrees, eoa_txns, ca_degrees, ca_txns):
    # Insert Node Feature
    for fp in tqdm(chunk_files):                                                              
        insert_node_data(fp, eoa_degrees, eoa_txns, ca_degrees, ca_txns)    

def mp_insert(files, eoa_degrees, eoa_txns, ca_degrees, ca_txns):
    # Insert Node Feature (Multiprocessing)
    length = len(files)//50
    chunks = [files[i:i+length] for i in range(0, len(files), length)]
    pool = Pool(len(chunks))
    for i in range(len(chunks)):
        pool.apply_async(insert, args=(chunks[i], eoa_degrees, eoa_txns, ca_degrees, ca_txns, ))
    pool.close()
    pool.join()
