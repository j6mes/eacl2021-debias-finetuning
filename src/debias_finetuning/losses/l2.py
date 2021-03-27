import logging
from copy import deepcopy
from allennlp.data import DataLoader
from torch import nn

logger = logging.getLogger(__name__)


class L2(object):

    def __init__(self, dataset: DataLoader):
        self.dataset = dataset

    def consolidate(self, model: nn.Module, cuda_device:int):
        logger.info("Consolidate model")
        self.cuda_device = cuda_device
        self.model = model
        if not hasattr(self, "original_params"):
            self.original_params = {n: deepcopy(p) for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and p in self.original_params:
                _loss = (self.original_params[n] - p) ** 2
                loss += _loss.sum()
        return loss
