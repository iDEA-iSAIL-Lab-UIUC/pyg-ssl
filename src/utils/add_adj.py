import torch
from torch_geometric.typing import SparseTensor


def add_adj_t(dataset):
    batch = dataset.data
    if not hasattr(batch, 'adj_t'):
        value = torch.ones_like(batch.edge_index[0], dtype=torch.float)
        batch.adj_t = SparseTensor(
            row=batch.edge_index[1],
            col=batch.edge_index[0],
            value=value,
            sparse_sizes=(batch.x.shape[0], batch.x.shape[0]),
            is_sorted=True,
            trust_data=True,
        )
    return dataset