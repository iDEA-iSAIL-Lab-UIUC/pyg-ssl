from .base import Augmentor
import copy
import random
import torch
from src.augment.random_drop_node import delete_row_col
# from torch_geometric.utils.convert import to_scipy_sparse_matrix

class AugmentSubgraph(Augmentor):
    def __init__(self, is_x: bool = True, is_adj: bool = False, drop_percent=0.2):
        super().__init__()
        self.is_x = is_x
        self.is_adj = is_adj
        self.drop_percent = drop_percent

    def __call__(self, data, drop_percent=None):
        drop_percent = drop_percent if drop_percent else self.drop_percent
        data_tmp = copy.deepcopy(data)
        input_adj = data_tmp.adj_t.to_dense()
        # input_adj = from_scipy_sparse_matrix(data_tmp.edge_index)
        input_fea = data_tmp.x.squeeze(0)
        node_num = input_fea.shape[0]

        all_node_list = [i for i in range(node_num)]
        s_node_num = int(node_num * (1 - drop_percent))
        center_node_id = random.randint(0, node_num - 1)
        sub_node_id_list = [center_node_id]
        all_neighbor_list = []

        for i in range(s_node_num - 1):

            all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

            all_neighbor_list = list(set(all_neighbor_list))
            new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
            if len(new_neighbor_list) != 0:
                new_node = random.sample(new_neighbor_list, 1)[0]
                sub_node_id_list.append(new_node)
            else:
                break

        drop_node_list = sorted([i for i in all_node_list if i not in sub_node_id_list])

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
        aug_input_adj = delete_row_col(input_adj, drop_node_list)

        data_tmp.x = aug_input_fea
        data_tmp.adj_t = aug_input_adj.to_sparse()

        return data_tmp
