import copy

import torch
from torch_geometric.data import Data

from .base import Augmentor


class ShuffleNode(Augmentor):
    r"""Randomly shuffle nodes."""
    def __init__(self):
        super().__init__()

    def __call__(self, data: Data):
        data_tmp = copy.deepcopy(data)
        idx_tmp = torch.randperm(len(data.x))
        data_tmp.x = data.x[idx_tmp]
        return data_tmp
