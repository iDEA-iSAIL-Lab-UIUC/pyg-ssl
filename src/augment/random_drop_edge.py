from .base import Augmentor
from src.data import HomoData
import copy
import random
import torch


class RandomDropEdge(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False, drop_percent=0.2):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj
        self.drop_percent = drop_percent

    def __call__(self, data: HomoData, drop_percent=None):
        drop_percent = drop_percent if drop_percent else self.drop_percent
        data_tmp = copy.deepcopy(data)
        percent = self.drop_percent / 2
        row_idx, col_idx = data.x.nonzero().T

        index_list = []
        for i in range(len(row_idx)):
            index_list.append((row_idx[i], col_idx[i]))

        edge_num = int(len(row_idx) / 2)  # 9228 / 2
        add_drop_num = int(edge_num * percent / 2)
        aug_adj = copy.deepcopy(data_tmp.adj_t.to_dense())

        edge_idx = [i for i in range(edge_num)]
        drop_idx = random.sample(edge_idx, add_drop_num)

        for i in drop_idx:
            aug_adj[index_list[i][0]][index_list[i][1]] = 0
            aug_adj[index_list[i][1]][index_list[i][0]] = 0

        node_num = data_tmp.x.shape[0]
        l = [(i, j) for i in range(node_num) for j in range(i)]
        add_list = random.sample(l, add_drop_num)

        for i in add_list:
            aug_adj[i[0]][i[1]] = 1
            aug_adj[i[1]][i[0]] = 1

        data_tmp.adj = aug_adj.to_sparse()
        return data_tmp
