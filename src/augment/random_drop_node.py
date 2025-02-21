from .base import Augmentor
import copy
import random
import torch


class RandomDropNode(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False, drop_percent=0.2):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj
        self.drop_percent = drop_percent
        
    def __call__(self, data, drop_percent=None):
        drop_percent = drop_percent if drop_percent else self.drop_percent
        data_tmp = copy.deepcopy(data)
        input_adj = torch.tensor(data_tmp.adj_t.to_dense().tolist())
        input_fea = data_tmp.x.squeeze(0)

        node_num = input_fea.shape[0]
        drop_num = int(node_num * drop_percent)  # number of drop nodes
        all_node_list = [i for i in range(node_num)]

        drop_node_list = sorted(random.sample(all_node_list, drop_num))

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
        aug_input_adj = delete_row_col(input_adj, drop_node_list)

        data_tmp.x = aug_input_fea
        data_tmp.adj_t = aug_input_adj.to_sparse()

        return data_tmp


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]
    return out