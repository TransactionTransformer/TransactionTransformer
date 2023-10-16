import os
import json
from ..dataset.fetch import gather, format_transactions

def download_node(node, dirpath):
    """Fetch Node Transactions using Etherscan API."""

    unformatted = gather(node)
    formatted = format_transactions(unformatted)
    
    os.makedirs(os.path.join(dirpath, 'frontier'), exist_ok=True)
    filepath = os.path.join(dirpath, 'frontier', f'{node}.json')
    with open(filepath, 'w') as f:
        f.write(json.dumps(formatted, indent=4))
    return formatted