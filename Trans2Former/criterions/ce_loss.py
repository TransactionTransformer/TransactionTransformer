from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("ce_loss", dataclass=FairseqDataclass)
class PretrainCELoss(FairseqCriterion):
    """
    Implementation for the cross entropy log loss used in edgeformer model training.
    """
    def __init__(self, task):
        super().__init__(task)
        self.criterion = CrossEntropyLoss()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
            
        logits = model(**sample["net_input"])
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])
        preds = torch.argmax(torch.sigmoid(logits), dim=1)

        mask = ~torch.isnan(targets)
        loss = self.criterion(logits, targets[mask].long())

        logging_output = {
            "loss": loss.data,
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsubgraphs": sample_size,
            "nsentences": sample_size,
            "ncorrect": (preds == targets).sum(), 
        }
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs retruned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
@register_criterion("ce_loss_with_flag", dataclass=FairseqDataclass)
class PretrainCELossWithFlag(PretrainCELoss):
    """
    Implementation for the cross entropy loss used in edgeformer model training
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)

        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[:logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)
        loss = F.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduce="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsubgraphs": sample_size,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(), 
        }
        return loss, sample_size, logging_output
    