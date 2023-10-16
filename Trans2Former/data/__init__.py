DATASET_REGISTRY = {}

def register_dataset(name: str):
    def register_dataset_func(func):
        DATASET_REGISTRY[name] = func()
    return register_dataset_func


def lookup_collator(dataset_source, task):
    if dataset_source == 'pkl' and task == 'pretrain':
        from .collator_token_pretrain import collator
        return collator
    elif dataset_source == 'pkl' and task == 'finetune':
        from .collator_token_finetune import collator_finetune
        return collator_finetune