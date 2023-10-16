# Transaction Transformer: Exploiting Internal Transactions for Graph Classification on Ethereum Blockchain

This is the anonymous respository for WWW24 paper:

> Transaction Transformer: Exploiting Internal Transactions for Graph Classification on Ethereum Blockchain

## Requirements and Installation

#### Setup with Conda

```
bash install.sh
```

## Usage

To build the pretrain model, one should

1. Download Block Transaction, Internal Transaction, ERC20 Transaction, ERC721 Transaction from [xblock](https://xblock.pro/xblock-eth.html)
2. Change *data_dir* in *build_dataset.py* to the downloaded xblock data directory.
3. Download our preprocessed summarization data, see [instruction](./dataset/preprocess/total/summary_data.sh)
4. To build subgraphs with the given pretrained addresses,
    ```
    python build_dataset.py
    ```
5. Pretrain the model
    ```
    cd example;
    bash pretrain.sh
    ```

To finetune the model with downstream tasks, one should

1. To build subgraphs for downstream tasks, go to dataset and run with specific task name, (i.e, phish-hack)
    ```
    python build_dataset.py phish-hack
    ```
2. Change to example directory, and change variable *category* in *finetune.sh*, and run
    ```
    bash finetune.sh
    ```