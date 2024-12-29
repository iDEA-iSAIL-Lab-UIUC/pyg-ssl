from .base import Augmentor
from src.data import HomoData
import copy
import torch
from scipy.linalg import fractional_matrix_power
from torch.linalg import inv

class ComputePPR(Augmentor):
    def __init__(self, alpha=0.2, self_loop=True):
        super().__init__()
        self.alpha = alpha
        self.self_loop = self_loop

    def __call__(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj_t.to_dense()
        device = a.device
        if self.self_loop:
            a = torch.eye(a.shape[0], device=device) + a
        d = torch.diag(torch.sum(a, 1)).cpu()
        dinv = torch.from_numpy(fractional_matrix_power(d, -0.5)).to(device)
        at = torch.matmul(torch.matmul(dinv, a), dinv)
        aug_adj = self.alpha * inv((torch.eye(a.shape[0], device=device) - (1 - self.alpha) * at))
        data_tmp.adj_t = aug_adj.to_sparse()
        return data_tmp


class ComputeHeat(Augmentor):
    def __init__(self, t=5, self_loop=True):
        super().__init__()
        self.t = t
        self.self_loop = self_loop

    def __call__(self, data: HomoData):
        data_tmp = copy.deepcopy(data)
        a = data_tmp.adj_t.to_dense()
        device = a.device
        if self.self_loop:
            a = torch.eye(a.shape[0], device=device) + a
        d = torch.diag(torch.sum(a, 1))
        aug_adj = torch.exp(self.t * (torch.matmul(a, inv(d)) - 1))
        data_tmp.adj_t = aug_adj.to_sparse()
        return data_tmp
