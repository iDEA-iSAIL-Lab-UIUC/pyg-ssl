from .base import BaseMethod
from torch_geometric.typing import *
from src.augment import RandomMask, ShuffleNode
from typing import Optional, Callable, Union
from src.typing import AugmentType
from .utils import AvgReadout
from src.losses import NegativeMI
from src.loader import AugmentDataLoader
from torch_geometric.nn.models import GCN
from torch_geometric.nn import GCNConv
import torch


class GraphCL(BaseMethod):
    r"""
    TODO: add descriptions
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 corruption: AugmentType = RandomMask(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        super().__init__(encoder=encoder, data_augment=corruption, loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()
        self.data_augment = corruption
        self.loss_function = loss_function if loss_function else NegativeMI(hidden_channels)

    def forward(self, batch):
        pos_batch, neg_batch, neg_batch2 = batch
        h_pos = self.encoder(pos_batch, pos_batch.adj_t)
        if self.augment_type == 'edge':
            # h_neg1 = self.encoder(pos_batch, neg_batch.adj_t.to(pos_batch.batch.device))
            # h_neg2 = self.encoder(pos_batch, neg_batch2.adj_t.to(pos_batch.batch.device))
            h_neg1 = self.encoder(pos_batch, neg_batch.edge_index)
            h_neg2 = self.encoder(pos_batch, neg_batch2.edge_index)
        elif self.augment_type == 'mask':
            # h_neg1 = self.encoder(neg_batch, pos_batch.adj_t)
            # h_neg2 = self.encoder(neg_batch2, pos_batch.adj_t)
            h_neg1 = self.encoder(neg_batch, pos_batch.edge_index)
            h_neg2 = self.encoder(neg_batch2, pos_batch.edge_index)
        elif self.augment_type == 'node' or self.augment_type == 'subgraph':
            # h_neg1 = self.encoder(neg_batch, neg_batch.adj_t.to(pos_batch.batch.device))
            # h_neg2 = self.encoder(neg_batch2, neg_batch2.adj_t.to(pos_batch.batch.device))
            h_neg1 = self.encoder(neg_batch, neg_batch.edge_index)
            h_neg2 = self.encoder(neg_batch2, neg_batch2.edge_index)
        else:
            assert False
        loss = self.get_loss(h_pos, h_neg1, h_neg2, pos_batch)
        return loss

    def apply_data_augment_offline(self, dataloader):
        batch_list = []
        for i, batch in enumerate(dataloader):
            batch_aug = self.data_augment(batch)
            batch_aug2 = self.data_augment(batch)
            batch_list.append((batch, batch_aug, batch_aug2))
        new_loader = AugmentDataLoader(batch_list=batch_list)
        return new_loader

    def apply_data_augment(self, batch):
        raise NotImplementedError

    def apply_emb_augment(self, h_pos):
        raise NotImplementedError

    def get_loss(self, h_pos, h_neg1, h_neg2, pos_batch):
        s1 = self.readout(h_neg1, keepdim=True)
        s1 = self.sigmoid(s1)
        s2 = self.readout(h_neg2, keepdim=True)
        s2 = self.sigmoid(s2)
        s1 = s1.expand_as(h_pos)
        s2 = s2.expand_as(h_pos)
        augmentation = ShuffleNode()
        neg_batch3 = augmentation(pos_batch).to(self._device)
        h_neg = self.encoder(neg_batch3, pos_batch.adj_t)

        loss1 = self.loss_function(x=s1, y=h_pos, x_ind=s1, y_ind=h_neg)
        loss2 = self.loss_function(x=s2, y=h_pos, x_ind=s2, y_ind=h_neg)
        return loss1 + loss2

    def get_embs(self, data):
        return self.encoder(data, data.adj_t).detach()


class GraphCLEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 num_layers=1,
                 bias=True):
        super(GraphCLEncoder, self).__init__()
        self.dim_out = hidden_channels
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(hidden_channels))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self._weights_init(m)
        self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=hidden_channels, bias=True)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, batch, edge_index, is_sparse=True):
        x = batch.x
        adj = batch.adj_t.to_dense().to(x.device)
        seq_fts = self.fc(x)
        if is_sparse:
            out = torch.mm(adj, torch.squeeze(seq_fts, 0))
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)