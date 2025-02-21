"""TODO: create a separate file for each class."""
import copy
import random
from scipy.linalg import fractional_matrix_power
import torch
from torch.linalg import inv
from torch_geometric.data import Data
from .base import Augmentor

class ComputePPR(Augmentor):
    def __init__(self, alpha=0.2, self_loop=True):
        super().__init__()
        self.alpha = alpha
        self.self_loop = self_loop

    def __call__(self, data):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj
        if self.self_loop:
            a = torch.eye(a.shape[0]) + a
        d = torch.diag(torch.sum(a, 1))
        dinv = torch.from_numpy(fractional_matrix_power(d, -0.5))
        at = torch.matmul(torch.matmul(dinv, a), dinv)
        data_tmp.adj = self.alpha * inv((torch.eye(a.shape[0]) - (1 - self.alpha) * at))
        return data_tmp


class ComputeHeat(Augmentor):
    def __init__(self, t=5, self_loop=True):
        super().__init__()
        self.t = t
        self.self_loop = self_loop

    def __call__(self, data):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj
        if self.self_loop:
            a = torch.eye(a.shape[0]) + a
        d = torch.diag(torch.sum(a, 1))
        data_tmp.adj = torch.exp(self.t * (torch.matmul(a, inv(d)) - 1))
        return data_tmp
