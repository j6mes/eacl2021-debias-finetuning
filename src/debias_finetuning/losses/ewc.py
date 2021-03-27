from copy import deepcopy
from typing import Dict, Any
import logging
import torch
from allennlp.data import DataLoader
from allennlp.nn.util import get_device_of, move_to_device
from allennlp.training import EpochCallback, GradientDescentTrainer
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data

from debias_finetuning.losses.util import variable

logger = logging.getLogger()


class EWC(object):
    def __init__(self, dataset: DataLoader):
        self.dataset = dataset

    def consolidate(self, model: nn.Module, cuda_device:int):
        logger.info("Consolidate model")
        self.cuda_device = cuda_device
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        device = None
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
            if device is None:
                device = get_device_of(p)

        iterator = iter(self.dataset)
        is_tr = self.model.training
        self.model.train()
        for batch in tqdm(iterator,desc="Computing fischer matrices"):
            self.model.zero_grad()
            batch = move_to_device(batch, device)
            output_dict = self.model.old_forward(**batch)
            loss = output_dict.get("loss")
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset.dataset.instances)
                    elif p.grad is None:
                        logger.warning("Parameter {} is none".format(n))
                        precision_matrices[n].data += torch.zeros_like(p.data)
        if not is_tr:
            self.model.eval()
        return {n: p for n, p in precision_matrices.items()}

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

