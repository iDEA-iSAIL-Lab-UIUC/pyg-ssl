from .base import Augmentor
from src.data import HomoData
import copy
import random
import torch


class RandomMask(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False, drop_percent=0.2):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj
        self.drop_percent = drop_percent

    def __call__(self, data: HomoData, drop_percent=None):
        drop_percent = drop_percent if drop_percent else self.drop_percent
        data_tmp = copy.deepcopy(data)
        if self.is_x:
            mask_num = int(data_tmp.num_nodes * drop_percent)
            node_idx = [i for i in range(data_tmp.num_nodes)]
            mask_idx = random.sample(node_idx, mask_num)
            zeros = torch.zeros_like(data_tmp.x[0])
            for j in mask_idx:
                data_tmp.x[j, :] = zeros
        if self.is_adj:
            raise NotImplementedError
        return data_tmp
