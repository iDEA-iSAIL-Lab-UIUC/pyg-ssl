import copy
import torch
from .base import BaseMethod
import numpy as np
# from src.augment import DataAugmentation, AugNegDGI, AugPosDGI
from src.loader import Loader, FullLoader
from .base import BaseMethod
from .utils import EMA, update_moving_average
from torch_geometric.nn.models import GCN
from torch_geometric.typing import *
import torch.nn.functional as F
from src.loader import AugmentDataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BGRL(BaseMethod):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, student_encoder: torch.nn.Module, teacher_encoder: torch.nn.Module, data_augment = None, pred_dim=None):
        super().__init__(encoder=student_encoder, data_augment=data_augment, loss_function=loss_fn)
        self.encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.data_augment = data_augment
        self.loss_function = loss_fn
        set_requires_grad(self.teacher_encoder, False)
        rep_dim = self.encoder.hidden_channels
        if pred_dim==None:
            pred_dim = rep_dim*2
        # self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_dim), nn.PReLU(), nn.Linear(pred_dim, rep_dim))
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, rep_dim))
        self.student_predictor.apply(init_weights)

    # def forward(self, x1: Tensor, x2: Tensor, adj1: Adj, adj2: Adj, is_sparse: bool = True):
    def forward(self, batch):
        batch_1, batch_2 = batch
        h_1 = self.encoder(batch_1, batch_1.edge_index)
        h_2 = self.encoder(batch_2, batch_2.edge_index)

        h_1_pred = self.student_predictor(h_1)
        h_2_pred = self.student_predictor(h_2)

        with torch.no_grad():
            v_1 = self.teacher_encoder(batch_1, batch_1.edge_index)
            v_2 = self.teacher_encoder(batch_2, batch_2.edge_index)
        loss1 = self.loss_function(h_1_pred, v_2.detach())
        loss2 = self.loss_function(h_2_pred, v_1.detach())

        loss = loss1 + loss2
        return loss.mean()
    
    def apply_data_augment_offline(self, dataloader):
        batch_list = []
        for i, batch in enumerate(dataloader):
            batch = batch.to(self._device)
            batch_aug_1 = self.data_augment["augment_1"](batch)
            batch_aug_2 = self.data_augment["augment_2"](batch)
            batch_list.append((batch_aug_1, batch_aug_2))
        new_loader = AugmentDataLoader(batch_list=batch_list)
        return new_loader
    
    def get_embs(self, data):
        return self.encoder(data, data.edge_index).detach()


class BGRLEncoder(nn.Module):

    def __init__(self, in_channel, hidden_channels, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels[-1]
        self.conv1 = GCNConv(in_channel, hidden_channels[0])
        self.bn1 = nn.BatchNorm1d(hidden_channels[0], momentum = 0.01)
        self.prelu1 = nn.PReLU()
        self.num_layer = len(hidden_channels) 
        self.conv_list, self.bn_list, self.act_list = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        for i in range(self.num_layer-1):
            self.conv_list.append(GCNConv(hidden_channels[i],hidden_channels[i+1]))
            self.bn_list.append(nn.BatchNorm1d(hidden_channels[i+1], momentum = 0.01))
            self.act_list.append(nn.PReLU())

    def forward(self, data, edge_index, edge_weight=None):
        x = data.x
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        for i in range(self.num_layer-1):
            x = self.conv_list[i](x, edge_index, edge_weight=edge_weight)
            x = self.act_list[i](self.bn_list[i](x))
        return x


class BGRLEncoder_old(torch.nn.Module):
    def __init__(self,
                 in_channel: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 num_layers=1):
        super(BGRLEncoder_old, self).__init__()
        self.hidden_channels=hidden_channels
        self.gcn = GCN(in_channels=in_channel, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
        self.act = act
        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, batch, edge_index, is_sparse=True):
        edge_weight = batch.edge_weight if "edge_weight" in batch else None
        return self.act(self.gcn(x=batch.x, edge_index=edge_index, edge_weight=edge_weight))
    