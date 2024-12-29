from .base import Augmentor
from src.data import HomoData
import copy
import random
import torch


class RandomMaskChannel(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False, drop_percent=0.2):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj
        self.drop_percent = drop_percent

    def __call__(self, data: HomoData, drop_percent=None):
        drop_percent = drop_percent if drop_percent else self.drop_percent
        data_tmp = copy.deepcopy(data)
        device = data_tmp.x.device
        if self.is_x:
            feat_mask = torch.FloatTensor(data_tmp.x.shape[1]).uniform_() > drop_percent
            feat_mask = feat_mask.to(device)
            data_tmp.x = feat_mask*data_tmp.x
        if self.is_adj:
            raise NotImplementedError
        return data_tmp
