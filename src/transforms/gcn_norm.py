import torch
from torch_sparse import SparseTensor, fill_diag, mul, set_diag
from torch_sparse import sum as sparsesum
from torch_scatter import scatter_add

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops

# based on torch_geometric.nn.conv.gcn_conv & torch_geometric.transforms.gcn_norm.py
# from torch_geometric.transforms.gcn_norm import GCNNorm


@functional_transform('gcn_norm_ssl')
class GCNNorm(BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """
    def __init__(self, add_self_loops: int = 1):
        self.add_self_loops = add_self_loops

    def __call__(self, data):
        assert 'edge_index' in data or 'adj_t' in data

        if 'edge_index' in data:
            edge_weight = data.edge_attr
            if 'edge_weight' in data:
                edge_weight = data.edge_weight
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index, edge_weight, data.num_nodes,
                add_self_loops=self.add_self_loops)
        if 'adj_t' in data:  # It is possible that the data has both edge_index and adj_t
            data.adj_t = gcn_norm(data.adj_t, add_self_loops=self.add_self_loops)

        return data


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loops=1., flow="source_to_target", dtype=None):
    r"""Directly use add_self_loop to indicate the number of self_loops."""

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        # adj_t = fill_diag(adj_t, add_self_loops)  this seems incorrect
        diag_val = adj_t.get_diag() + add_self_loops
        adj_t = set_diag(adj_t, diag_val)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, add_self_loops, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
