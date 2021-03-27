import logging
import torch
from allennlp.training import EpochCallback, GradientDescentTrainer
from torch.autograd import Variable
from typing import Dict, Any

logger = logging.getLogger(__name__)


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class CallLossCallback(EpochCallback):
    def __init__(self, ewc):
        self.model = None
        self.ewc = ewc

    def __call__(self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int) -> None:
        logger.info("Consolidate model")
        self.ewc.consolidate(trainer.model, trainer.cuda_device)