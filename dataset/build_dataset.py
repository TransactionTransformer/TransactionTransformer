from multiprocessing import Pool
import sys
import logging
import utils
import glob
import fcntl
from collections import defaultdict
import os
import pickle as pkl
import random
from tqdm import tqdm
import torch

data_dir = "<Path to xblock data>"
address_dir = "<Path to addresses data>"


def get_logger(section):
    logging.basicConfig(
        filename='{}.log'.format(section),
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger('build.log')


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
    from_address, to_address = '', ''
    fields = txn.split(',')
    if etype == 0:
        from_address, to_address = fields[3], fields[4]
        if fields[5] != 'None': # toCreate is not None
            to_address = fields[5]
    else:
        from_address, to_address = fields[1], fields[2]
    return from_address, to_address

def get_neighbors(txns: dict, address: str):
    """ 
    Get the neighbor of address, and 
    return a dict whose key is neighbor address while
    value is a set of string, each indicating address
    and txnIdx
    
    Parameters:
    -------------------------------------------------
    txns: addresses related txns, 
            dict, {txnIdx: {typeIdx: list of txns str}}
    address: the center address
            string

    Return:
    -------------------------------------------------
    neighbors: dict {address's neighbors: set(txnIdx)}
    """
    neighbors = defaultdict(set)
    for txnIdx, typetxns in txns.items():
        # address_txnIdx = '{}_{}'.format(address, txnIdx)
        for etype in range(4):
            if etype not in typetxns: continue
            for txn in typetxns[etype]:
                from_address, to_address = get_from_to_address(txn, etype)
                if etype == 0: # if external transaction add all neighbors
                    if from_address != address: neighbors[from_address].add(txnIdx if to_address == address else -1)
                    if to_address != address: neighbors[to_address].add(txnIdx if from_address == address else -1)
                    continue
                else: # other transaction only use direct neighbors
                    if address != from_address and address != to_address: continue 
                    if address == from_address and address == to_address: continue # ignore self-loop
                    neighbor = from_address if address == to_address else to_address
                    # if A,B both have txns with C, then A,B would have had the same sampled C
                    # if A have multiple txns with C, then A should have only one copy of sampled C
                    if not isinstance(neighbor, float) and neighbor != 'None' and neighbor != 'none' \
                        and neighbor != '0x0000000000000000000000000000000000000000':
                        neighbors[neighbor].add(txnIdx)
    return neighbors

def get_1hop_neighbors_(addressTxns):
    addresses_toextract = defaultdict(list)
    for center_address, txns in tqdm(addressTxns.items()):
        neighbors = get_neighbors(txns, center_address)
        for neighbor, txnIndices in neighbors.items():
            if neighbor == '': continue
            assert neighbor != center_address
            txnIndices = [t for t in txnIndices if t != -1]
            addresses_toextract[neighbor].append([center_address] + list(txnIndices))
    return addresses_toextract

def get_1hop_neighbors(addressTxns):
    keys = list(addressTxns.keys())
    chunks = [keys[i:i+len(keys)//10] for i in range(0,len(keys),len(keys)//10)]
    chunks = [{key:addressTxns[key] for key in chunk} for chunk in chunks]
    results = []
    pool = Pool(len(chunks))
    for i in range(len(chunks)):
        results.append(pool.apply_async(get_1hop_neighbors_, args=(chunks[i], )))
    pool.close()
    pool.join()
    addresses_toextract = defaultdict(list)
    for res in results:
        res = res.get()
        for neighbor, related_set in res.items():
            addresses_toextract[neighbor] += related_set
    return addresses_toextract


def load_gt64(loaded_data, MAX_LENGTH):
    # address: {idx: {typeidx: list of transaction}}}
    data = defaultdict(dict)
    for address, fp in loaded_data.items():
        hash2txnIdx = {}
        with open(fp, 'r') as f:
            lines = f.readlines()
            for l in lines:
                txn = l[:-1] if l[-1] == '\n' else l
                fields = txn.split(',')
                # get txnIdx
                if fields[2] not in hash2txnIdx:
                    hash2txnIdx[fields[2]] = len(hash2txnIdx)
                txnIdx = hash2txnIdx[fields[2]]
                # append txn
                if fields[-1] == '0':
                    data[address][txnIdx] = {0: [txn]}
                else:
                    etype = int(fields[-1])
                    if etype not in data[address][txnIdx]:
                        data[address][txnIdx][etype] = list()
                    data[address][txnIdx][etype].append(txn.split(',',3)[-1]) # ignore bn,ts,txnHash to save memory
    for address, txns in data.items():
        if len(txns) > MAX_LENGTH:
            reduce_txns = [txns[i] for i in range(MAX_LENGTH//4)]
            random_indices = random.sample(range(MAX_LENGTH//4, len(txns)-MAX_LENGTH//4), MAX_LENGTH//2)
            reduce_txns += [txns[idx] for idx in sorted(random_indices)]
            reduce_txns += [txns[i] for i in range(len(txns)-MAX_LENGTH//4, len(txns))]
            reduce_txns = {i: txns for i, txns in enumerate(reduce_txns)}
            data[address] = reduce_txns
    return data

def get_transaction_indices_(address_filepath, MAX_LENGTH=128, MIN_LENGTH=4):
    # read address indices file and get corresponding transactions from 'blocks' files
    # limited by MAX_LENGTH
    if not os.path.exists(address_filepath): 
        logger.info('Missing {}'.format(address_filepath))
        return dict() # return empty
    with open(address_filepath, 'r') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        lines = f.readlines()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        lines = [l[:-1] if l[-1] == '\n' else l for l in lines]
        lines = [l for l in lines if l != '']
        txns = []
        for l in lines:
            section_name, section_txns = l.split(':')
            txns += [(txn, section_name) for txn in section_txns.split('|')]
        if len(txns) < MIN_LENGTH:
            return {}
        elif MIN_LENGTH <= len(txns) <= MAX_LENGTH:
            sections = defaultdict(list)
            for txn, section_name in txns:
                sections[section_name].append(txn)
            return sections
        else:
            selected_txns = txns[:MAX_LENGTH//4]
            randindices = random.sample(range(MAX_LENGTH//4, len(txns)-MAX_LENGTH//4), MAX_LENGTH//2)
            for i in sorted(randindices):
                selected_txns.append(txns[i])
            selected_txns += txns[-MAX_LENGTH//4:]
            sections = defaultdict(list)
            for txn, section_name in selected_txns:
                sections[section_name].append(txn)
            return sections


def get_transaction_indices(addresses, MAX_LENGTH, MIN_LEGNTH, pid):
    """ Read the address indices file, and seperate their txns indices into sections.
    The return value, `results` would be a dictionary whose key is `section` and value is 
    a list of items consisting of ([8 integers], 0xaddress, txnIdx), where 8 integers 
    indicate the rowNumber in 4 types of files

    """
    # address_dir = '/localdata_ssd/lifan/datasets/xblock/addresses'
    results = defaultdict(list)
    for i, address in enumerate(addresses):
        if len(addresses) < 10 or i % (len(addresses) // 10) == 0:
            logger.info('Process {}: {}/{}'.format(pid, i, len(addresses)))
        if len(address) < 6: continue # avoid strange address
        address_filepath = os.path.join(address_dir, address[2], address[3], address[4], address[5], "{}.txt".format(address))
        sections = get_transaction_indices_(address_filepath, MAX_LENGTH, MIN_LEGNTH)
        prev_idx = 0
        for section, txns in sections.items():
            results[section] += [tuple(list(map(int, txn.split(','))) + [address, prev_idx+idx]) for idx, txn in enumerate(txns)]
            prev_idx += len(txns)
    return results

def read_file_by_lines_(fp, type, rownum_pairs):
    results = defaultdict(dict)
    with open(fp, 'r') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        current_line = 0
        current_idx = 0
        line = f.readline()
        result = []
        while(current_idx < len(rownum_pairs) and line != '' and line is not None):
            current_line += 1
            pair = rownum_pairs[current_idx]
            line = f.readline()
            if pair[1] >= current_line >= pair[0]:
                if line == '' or line is None: 
                    pass
                else:
                    if line[-1] == '\n': line = line[:-1]
                    if type != 0: line = line.split(',',3)[-1]
                    result.append(line+','+str(pair[0])+','+str(type))
                if current_line == pair[1]:
                    while(current_idx < len(rownum_pairs) and 
                            pair[0] == rownum_pairs[current_idx][0] and
                           pair[1] == rownum_pairs[current_idx][1]): # which means several addresses using this transaction
                        address = rownum_pairs[current_idx][2]
                        txnIdx = rownum_pairs[current_idx][3]
                        results[address][txnIdx] = {type:result}
                        current_idx += 1
                    # reset
                    result = []
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return results

def read_files_by_lines(file_indices, pid):
    """ file_indices consists of [file path, type idx, indices], where `indices` is
    a sorted list of ([2 integers], 0xaddress, txnIdx). So that the coordinator can
    fetch all the related data in one-time reading without repeatedly iterating
    from begining for every address.
    """
    results = defaultdict(dict)
    for idx, (fp, type, indices) in enumerate(file_indices):
        if len(file_indices) < 10 or idx % (len(file_indices) // 10) == 0:
            logger.info('Process {}: {}/{}'.format(pid, idx, len(file_indices)))
        temp_results = read_file_by_lines_(fp, type, indices)
        for address in temp_results:
            if address not in results:
                results[address] = temp_results[address]
            else:
                for txnIdx in temp_results[address]:
                    if txnIdx not in results[address]:
                        results[address][txnIdx] = temp_results[address][txnIdx]
                    else:
                        results[address][txnIdx].update(temp_results[address][txnIdx])
    return results


def fetch_data(chunk, chunkidx, save_fn, chunksize=1_000_000, MAX_LENGTH=128, MIN_LENGTH=4):
    logger.info('=======================Chunk {} start======================='.format(chunkidx))
    
    # Get indices
    if len(chunk) > 1000:
        subchunksize = len(chunk) // 100
        subchunks = [chunk[i:i+subchunksize] for i in range(0, len(chunk), subchunksize)]
    else:
        subchunks = [chunk]
    pool = Pool(len(subchunks))
    results=[]
    for i in range(len(subchunks)):
        results.append(pool.apply_async(get_transaction_indices, args=(subchunks[i], MAX_LENGTH, MIN_LENGTH, i, )))
    pool.close()
    pool.join()
    logger.info('Address file readed, now coordinate')

    # coordinate
    section2indices = defaultdict(list)
    for res in results:
        rs = res.get()
        for section, txns in rs.items():
            section2indices[section]+=txns
    del results
    
    # flatten
    file2indices = list()
    for section, txns in section2indices.items():
        for typeidx, type in enumerate(['external', 'internal', 'erc20', 'erc721']):
            filepath = os.path.join(data_dir, section, '{}_{}Transaction.csv'.format(section, type))          
            if not os.path.exists(filepath): continue
            typetxns = [txn[typeidx*2:(typeidx*2)+2]+txn[-2:] for txn in txns if txn[typeidx*2:(typeidx*2)+2] != [-1,-1] and txn[typeidx*2:(typeidx*2)+2] != (-1,-1)]
            typetxns = sorted(typetxns, key=lambda x: x[0])
            if len(typetxns) > 0:
                file2indices.append([filepath, typeidx, typetxns])
    logger.info('Coordinate read blocks file'.format(chunkidx))

    # multiprocessing reading
    length = len(file2indices)
    subchunks = [file2indices[i:i+length // 100] for i in range(0, length, length // 100)]
    pool2 = Pool(len(subchunks))
    results = []
    for i in range(len(subchunks)):
        results.append(pool2.apply_async(read_files_by_lines, args=(subchunks[i], i,)))
    pool2.close()
    pool2.join()
    data = {}
    for res in results:
        res = res.get()
        for address in res:
            if address not in data: data[address] = res[address]
            else:
                for txnIdx in res[address]:
                    if txnIdx not in data[address]:
                        data[address][txnIdx] = res[address][txnIdx]
                    else:
                        data[address][txnIdx].update(res[address][txnIdx])
    del results

    # save
    with open(save_fn, 'wb') as f:
        pkl.dump(data, f)
    logger.info('Write data file')

    return data


def fetch_full_process(addresses, 
                    LENGTH_RANGE=dict(
                        ONEHOP_MAX=128, ONEHOP_MIN=4, TWOHOP_MAX=16, TWOHOP_MIN=4), 
                    name='',
                    dataset_type='pretrain'):
    """Get Ego Subgraph and its corresponding NCP Subgraph"""
    chunksize = 1_000_000
    chunks = [addresses[i:i+chunksize] for i in range(0,len(addresses),chunksize)]
    for chunkidx, chunk in enumerate(chunks):
        logger.info(f'------------------Preprocessing {chunkidx}th 1_000_000 nodes-----------------')
        data = fetch_data(chunk, chunkidx,
               save_fn='./preprocess/temp/fetch_data_{}.pkl'.format(chunkidx if name=='' else name),
               chunksize=chunksize, MAX_LENGTH=LENGTH_RANGE["ONEHOP_MAX"], MIN_LENGTH=LENGTH_RANGE['ONEHOP_MIN'])
        
        # get neighbors
        logger.info(f'---------Get {chunkidx}th onehop neighbors transactions---------')
        addresses_toextract = get_1hop_neighbors(data)
        center2neighbor = defaultdict(set)
        for neighbor in addresses_toextract:
            for cl in addresses_toextract[neighbor]:
                center2neighbor[cl[0]].add(neighbor)
        for center, neighbors in center2neighbor.items():
            if len(neighbors) > LENGTH_RANGE["ONEHOP_MAX"]:
                center2neighbor[center] = set(random.sample(list(neighbors), LENGTH_RANGE["ONEHOP_MAX"]))
        at = defaultdict(list)
        for center, neighbors in center2neighbor.items():
            for neighbor in neighbors:
                d = {cl[0]: i for i, cl in enumerate(addresses_toextract[neighbor])}
                if center in d:
                    at[neighbor].append(addresses_toextract[neighbor][d[center]])
        addresses_toextract = at
        del center2neighbor
        with open('./preprocess/temp/neighbor2center_{}.pkl'.format(chunkidx if name=='' else name), 'wb') as f:
            pkl.dump(addresses_toextract, f)
        del data

        # fetch neighbors' txns
        logger.info(f'---------Get {chunkidx}th twohop neighbors transactions---------')
        neighbors = list(addresses_toextract.keys())
        for subchunkidx, subchunk in enumerate([neighbors[i:i+chunksize] for i in range(0,len(neighbors),chunksize)]):
            fetch_data(subchunk, subchunkidx,
                save_fn='./preprocess/temp/2hop/neighbor_{}-{}.pkl'.format(chunkidx if name=='' else name, subchunkidx),
                chunksize=chunksize, MAX_LENGTH=LENGTH_RANGE['TWOHOP_MAX'], MIN_LENGTH=LENGTH_RANGE['TWOHOP_MIN'])
        
        # load idx mapping
        logger.info(f'---------Index mapping {chunkidx}th onehop & twohop data---------')
        eoa2idx = pkl.load(open('./preprocess/total/eoa.pkl','rb'))
        ca2idx = pkl.load(open('./preprocess/total/ca.pkl','rb'))
        token2idx = pkl.load(open('./preprocess/total/token.pkl','rb'))
        func2idx = pkl.load(open('./preprocess/total/func.pkl','rb'))
        # mapping 1hop to a compact size
        fp = './preprocess/temp/fetch_data_{}.pkl'.format(chunkidx if name=='' else name)
        save_dir = './{}/{}/'.format(dataset_type, chunkidx if name=='' else name)
        utils.mapping(fp, save_dir, eoa2idx, ca2idx,token2idx, func2idx)
        # mapping 2hop to a compact size
        fps = glob.glob('./preprocess/temp/2hop/neighbor_{}-*.pkl'.format(chunkidx if name=='' else name))
        for fp in fps:
            utils.mapping_2hop(addresses_toextract, fp, save_dir, eoa2idx, ca2idx, token2idx, func2idx)

        # predefined neighbors for NCP
        addresses_toextract = pkl.load(open('./preprocess/temp/neighbor2center_{}.pkl'.format(chunkidx if name=='' else name),'rb'))
        logger.info(f'---------Get NCP pair {chunkidx}th---------')
        revert  = defaultdict(set)
        for neighbor, center_list in addresses_toextract.items():
            for cl in center_list:
                revert[cl[0]].add(neighbor)
        predefined_neighbors = dict()
        for center, neighbors in revert.items():
            predefined_neighbors[center] = random.sample(list(neighbors), 1)[0]
        with open('./preprocess/temp/{}/NCP_mapping.pkl'.format(chunkidx if name == '' else name),'wb') as f:
            pkl.dump(predefined_neighbors, f)
        predefined_neighbors = list(set(predefined_neighbors.values()))
        # fetch NCP nodes 1hop
        logger.info(f'---------Get NCP pair {chunkidx}th onehop neighbor---------')
        data = fetch_data(predefined_neighbors, chunkidx,
               save_fn='./preprocess/temp/fetch_data_NCP_{}.pkl'.format(chunkidx if name == '' else name),
               chunksize=chunksize, MAX_LENGTH=LENGTH_RANGE['ONEHOP_MAX'], MIN_LENGTH=LENGTH_RANGE['ONEHOP_MIN'])
        # NCP 2hop neighbors
        addresses_toextract = get_1hop_neighbors(data)
        center2neighbor = defaultdict(set)
        for neighbor in addresses_toextract:
            for cl in addresses_toextract[neighbor]:
                center2neighbor[cl[0]].add(neighbor)
        for center, neighbors in center2neighbor.items():
            if len(neighbors) > LENGTH_RANGE['ONEHOP_MAX']:
                center2neighbor[center] = set(random.sample(list(neighbors), LENGTH_RANGE['ONEHOP_MAX']))
        at = defaultdict(list)
        for center, neighbors in center2neighbor.items():
            for neighbor in neighbors:
                d = {cl[0]: i for i, cl in enumerate(addresses_toextract[neighbor])}
                if center in d:
                    at[neighbor].append(addresses_toextract[neighbor][d[center]])
        addresses_toextract = at
        del center2neighbor
        with open('./preprocess/temp/NCPneighbor2NCPcenter_{}.pkl'.format(chunkidx if name == '' else name), 'wb') as f:
            pkl.dump(addresses_toextract, f)
        del data
        logger.info(f'---------Get NCP pair {chunkidx}th twohop neighbors---------')
        NCP_neighbors = list(addresses_toextract.keys())
        for subchunkidx, subchunk in enumerate([NCP_neighbors[i:i+chunksize] for i in range(0,len(NCP_neighbors),chunksize)]):
            fetch_data(subchunk, subchunkidx,
                save_fn='./preprocess/temp/2hop/NCPneighbor_{}-{}.pkl'.format(chunkidx if name == '' else name, subchunkidx),
                chunksize=chunksize, MAX_LENGTH=LENGTH_RANGE['TWOHOP_MAX'], MIN_LENGTH=LENGTH_RANGE['TWOHOP_MIN'])
        # mapping NCP 1hop to a compact size
        logger.info(f'---------Index Mapping NCP {chunkidx}th  onehop & twohop data---------')
        fp = './preprocess/temp/fetch_data_NCP_{}.pkl'.format(chunkidx if name == '' else name)
        save_dir = './{}/{}/NCP/'.format(dataset_type, chunkidx if name == '' else name)
        utils.mapping(fp, save_dir, eoa2idx, ca2idx,token2idx, func2idx)
        # mapping 2hop to a compact size
        fps = glob.glob('./preprocess/temp/2hop/NCPneighbor_{}-*.pkl'.format(chunkidx if name == '' else name))
        for fp in fps:
            utils.mapping_2hop(addresses_toextract, fp, save_dir, eoa2idx, ca2idx, token2idx, func2idx)
        
        # Insert Node Data
        eoa_degrees = torch.load('./preprocess/total/eoa_degrees.pt')
        eoa_txns = torch.load('./preprocess/total/eoa_txns.pt') 
        ca_degrees = torch.load('./preprocess/total/ca_degrees.pt')
        ca_txns = torch.load('./preprocess/total/ca_txns.pt')
        node_files = glob.glob('./{}/{}/*/*.pkl'.format(dataset_type, chunkidx if name == '' else name))
        utils.mp_insert(node_files, eoa_degrees, eoa_txns, ca_degrees, ca_txns)
        node_files = glob.glob('./{}/{}/NCP/*/*.pkl'.format(dataset_type, chunkidx if name == '' else name))
        utils.mp_insert(node_files, eoa_degrees, eoa_txns, ca_degrees, ca_txns)


def build_pretrain_dataset():
    """Pretrain"""
    # read pretrained_addresses.txt
    addresses = open("./preprocess/total/pretrained_addresses.txt",'r').readline().split(',')
    # Set the upper bound and lower bound
    rangedict = dict(ONEHOP_MAX=128, ONEHOP_MIN=4, TWOHOP_MAX=16, TWOHOP_MIN=4)
    fetch_full_process(addresses, name='', dataset_type='pretrain', LENGTH_RANGE=rangedict)

def build_finetune_dataset(category):
    """Finetune"""
    # read specific finetune addresses
    label = pkl.load(open(f"./preprocess/total/{category}_label.txt",'r'))
    addresses = list(label.keys())
    # Set the upper bound and lower bound
    rangedict = dict(ONEHOP_MAX=128, ONEHOP_MIN=4, TWOHOP_MAX=16, TWOHOP_MIN=4)
    fetch_full_process(addresses, name='', dataset_type='finetune', LENGTH_RANGE=rangedict)

if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1: 
        build_pretrain_dataset()
    elif len(args) == 2:
        build_finetune_dataset(args[1])