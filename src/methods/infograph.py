import torch
from .base import BaseMethod
from .utils import AvgReadout
from src.losses import LocalGlobalLoss
from typing import Optional, Callable, Union
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
import torch.nn.functional as F


class InfoGraph(BaseMethod):
    r""" InfoGraph.

    Args:
        encoder (Optional[torch.nn.Module]): the encoder to be trained.
        hidden_channels (int): output dimension of the encoder.
        readout (Union[Callable, torch.nn.Module]): the readout function to obtain the summary emb. of the entire graph.
            (default: AvgReadout())
        corruption (OptAugment): data augmentation/corruption to generate negative node pairs.
            (default: ShuffleNode())
        loss_function (Optional[torch.nn.Module]): the loss function. If None, then use the NegativeMI loss.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 loss_function: Optional[torch.nn.Module] = LocalGlobalLoss(),
                 gamma=.1,
                 num_layers=1,
                 prior=False) -> None:

        super().__init__(encoder=encoder,  loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()
        self.gamma = gamma
        self.prior = prior
        self.embedding_dim = hidden_channels * num_layers
        self.local_d = MLP(self.embedding_dim)
        self.global_d = MLP(self.embedding_dim)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)
        self.loss_function = loss_function if loss_function else LocalGlobalLoss()
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        y, m = self.encoder(data)
        loss = self.get_loss(y, m, data)
        return loss

    def get_embs(self, dataset):
        ret = []
        y = []
        with torch.no_grad():
            for data in dataset:
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1))
                x, _ = self.encoder(data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = torch.FloatTensor(np.concatenate(ret, 0))
        y = torch.FloatTensor(np.concatenate(y, 0))
        return y, ret

    def apply_data_augment(self, batch):
        raise NotImplementedError

    def apply_emb_augment(self, h_pos):
        raise NotImplementedError

    def get_loss(self, y, m, data):
        g_enc = self.global_d(y)
        l_enc = self.local_d(m)

        measure = 'JSD'
        local_global_loss = self.loss_function(l_enc, g_enc, data.batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, GNN=GINConv):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.in_channels = in_channels

        for i in range(num_layers):
            if i:
                nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(),
                                Linear(hidden_channels, hidden_channels))
            else:
                nn = Sequential(Linear(in_channels, hidden_channels), ReLU(),
                                Linear(hidden_channels, hidden_channels))
            conv = GNN(nn)
            bn = torch.nn.BatchNorm1d(hidden_channels)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data):
        x = data.x
        if x is None:
            x = torch.ones((data.num_nodes, self.in_channels)).to(data.y.device)
        edge_index = data.edge_index
        batch = data.batch
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)