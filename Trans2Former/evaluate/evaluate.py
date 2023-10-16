import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
from os import path

sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))
from pretrain import load_pretrained_model

import logging

def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    # load checkpoint
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    model.to(torch.cuda.current_device())

    # load dataset
    split = args.split
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)
    y_p = torch.where(torch.sigmoid(y_pred) < 0.5, 0, 1)
    tp = torch.logical_and((y_p == y_true), (y_true==1)).sum()
    fn = torch.logical_and((y_p != y_true), (y_true==1)).sum()
    fp = torch.logical_and((y_p != y_true), (y_true==0)).sum()
    tn = torch.logical_and((y_p == y_true), (y_true==0)).sum()

    # evaluate pretrained models
    if 'auc' in args.metric:
        auc = roc_auc_score(y_true, y_pred)
        logger.info(f"auc: {auc}")
    if 'mae' in args.metric:
        mae = torch.mean(torch.abs(y_true-y_p))
        logger.info(f"mae: {mae}")
    if "acc" in args.metric:
        acc = (tp+tn)/(tp+tn+fp+fn)
        logger.info(f"acc: {acc}")
    if "recall" in args.metric:
        recall = tp/(tp+fn)
        logger.info(f"recall: {recall}")
    if "precision" in args.metric:
        precision = tp/(tp+fp)
        logger.info(f"precision: {precision}")
    if "f1" in args.metric:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        logger.info(f"f1: {f1}")

def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    parser.add_argument(
        "--is_evaluate",
        type=bool,
        default=True,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "None":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)

if __name__ == '__main__':
    main()
