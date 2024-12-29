import torch

import numpy as np
# from augment import DataAugmentation, AugNegDGI, AugPosDGI
from src.loader import Loader, FullLoader
from .base import BaseMethod
from torch_geometric.nn.models import GCN
from torch_geometric.typing import *
from torch_geometric.nn import GCNConv
import copy
from .utils import EMA, update_moving_average
import torch.nn.functional as F
from src.loader import AugmentDataLoader
import torch.nn as nn


class AFGRL(BaseMethod):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, student_encoder: torch.nn.Module, teacher_encoder: torch.nn.Module, data_augment=None, adj_ori = None, topk=8):
        super().__init__(encoder=student_encoder, data_augment=data_augment, loss_function=loss_fn)
        self.encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        set_requires_grad(self.teacher_encoder, False)
        rep_dim = self.encoder.hidden_channels
        pred_hid = rep_dim*2
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)
        self.data_augment = data_augment
        self.topk = topk
        self.adj_ori = adj_ori

    def forward(self, batch):
        student = self.encoder(batch, batch.edge_index)
        pred = self.student_predictor(student)

        with torch.no_grad():
            teacher = self.teacher_encoder(batch, batch.edge_index)
    
        adj_search = self.adj_ori
        student, teacher, pred = torch.squeeze(student), torch.squeeze(teacher), torch.squeeze(pred)
        ind, k = self.data_augment(adj_search, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk)

        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        loss = loss1 + loss2

        return loss.mean()

    def apply_data_augment_offline(self, dataloader):
        batch_list = []
        for i, batch in enumerate(dataloader):
            batch = batch.to(self._device)
            batch_list.append(batch)
        new_loader = AugmentDataLoader(batch_list=batch_list)
        return new_loader
    
    # def get_embs(self, x, edge_index):
    #     return self.encoder(x, edge_index)

    def get_embs(self, data):
        return self.encoder(data, data.edge_index).detach()


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


class AFGRLEncoder(nn.Module):

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