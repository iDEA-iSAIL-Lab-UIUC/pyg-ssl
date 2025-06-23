import time
import torch
from torch_geometric.loader import DataLoader
from src.methods import BaseMethod
from .base import BaseTrainer
from .utils import EarlyStopper
from typing import Union

from src.augment.gca_augments import drop_edge_weighted, drop_feature, drop_feature_weighted_2
from src.augment.gca_augments import degree_drop_weights, pr_drop_weights, evc_drop_weights
from src.augment.gca_augments import feature_drop_weights, feature_drop_weights_dense
from src.augment.gca_augments import compute_pr, eigenvector_centrality

from torch_geometric.utils import dropout_adj, degree, to_undirected


class GCATrainer(BaseTrainer):
    r"""
    TODO: 1. Add descriptions.
          2. Do we need to support more arguments?
    """
    def __init__(self,
                 method: BaseMethod,
                 data_loader: DataLoader,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 5000,
                 patience: int = 50,
                 drop_scheme: str = 'degree',
                 dataset_name: str = 'WikiCS',
                 device: Union[str, int] = "cuda:1",
                 save_root: str = "./ckpt"):
        super().__init__(method=method,
                         data_loader=data_loader,
                         save_root=save_root,
                         device=device)
        self.device = device

        self.drop_scheme = drop_scheme

        self.dataset_name = dataset_name

        self.optimizer = torch.optim.Adam(self.method.parameters(), lr, weight_decay=weight_decay)

        self.n_epochs = n_epochs
        self.patience = patience

        self.early_stopper = EarlyStopper(patience=self.patience)

    def train(self):
        self.method = self.method.to(self.device)
        for epoch in range(self.n_epochs):
            start_time = time.time()

            self.method.train()
            self.optimizer.zero_grad()

            for data in self.data_loader:
                # ---> 2. augment

                # ------> 2.1 drop_weights
                data = data.to(self.device)

                if self.drop_scheme == 'degree':
                    drop_weights = degree_drop_weights(data.edge_index).to(self.device)
                elif self.drop_scheme == 'pr':
                    drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(self.device)
                elif self.drop_scheme == 'evc':
                    drop_weights = evc_drop_weights(data).to(self.device)
                else:
                    drop_weights = None

                # ------> 2.2 feature_weights
                if self.drop_scheme == 'degree':
                    edge_index_ = to_undirected(data.edge_index)
                    node_deg = degree(edge_index_[1])
                    if self.dataset_name == 'WikiCS':
                        feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(self.device)
                    else:
                        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(self.device)
                elif self.drop_scheme == 'pr':
                    node_pr = compute_pr(data.edge_index)
                    if self.dataset_name == 'WikiCS':
                        feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(self.device)
                    else:
                        feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(self.device)
                elif self.drop_scheme == 'evc':
                    node_evc = eigenvector_centrality(data)
                    if self.dataset_name == 'WikiCS':
                        feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(self.device)
                    else:
                        feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(self.device)
                else:
                    feature_weights = torch.ones((data.x.size(1),)).to(self.device)


                def drop_edge(idx: int):
                    # global drop_weights

                    if idx == 1:
                        drop_edge_rate = 0.3
                    else:
                        drop_edge_rate = 0.4

                    if self.drop_scheme == 'uniform':
                        return dropout_adj(data.edge_index, p=drop_edge_rate)[0]
                    elif self.drop_scheme in ['degree', 'evc', 'pr']:
                        return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate, threshold=0.7)
                    else:
                        raise Exception(f'undefined drop scheme: {self.drop_scheme}')

                drop_feature_rate_1 = 0.1
                drop_feature_rate_2 = 0.0

                edge_index_1 = drop_edge(1)
                edge_index_2 = drop_edge(2)
                x_1 = drop_feature(data.x, drop_feature_rate_1)
                x_2 = drop_feature(data.x, drop_feature_rate_2)

                if self.drop_scheme in ['pr', 'degree', 'evc']:
                    x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
                    x_2 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_2)

                z1 = self.method(x_1, edge_index_1)
                z2 = self.method(x_2, edge_index_2)

                loss = self.method.loss(z1, z2, batch_size=1024)
                loss.backward()
                self.optimizer.step()

                end_time = time.time()
                print("Epoch {}: loss: {:.4f}, time: {:.4f}s".format(epoch, loss, end_time - start_time))

                self.early_stopper.update(loss)  # update the status
                if self.early_stopper.save:
                    self.save()
                if self.early_stopper.stop:
                    return

