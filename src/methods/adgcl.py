import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

from .base import Model
from nn.utils import AvgReadout, DiscriminatorMVGRL

from torch_geometric.typing import Tensor, Adj


class SUGRL(Model):
    r"""The full model to train the encoder.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        discriminator (torch.nn.Module): the discriminator for contrastive learning.
    """
    def __init__(self, encoder: torch.nn.Module):
        super().__init__(encoder=encoder)
        self.encoder_1 = encoder[0]
        self.encoder_2 = encoder[1]
        self.read = AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: Tensor, x_neg: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True,
                msk: Tensor = None, samp_bias1: Tensor = None, samp_bias2: Tensor = None):
        h_1 = self.encoder_1(x, adj, is_sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigmoid(c_1)
        h_2 = self.encoder_2(x, diff, is_sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigmoid(c_2)
        h_3 = self.encoder_1(x_neg, adj, is_sparse)
        h_4 = self.encoder_2(x_neg, diff, is_sparse)
        ret = self.discriminator(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)
        return ret, h_1, h_2

    def get_embs(self, x: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True, msk: Tensor = None):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()

    def get_embs_numpy(self, x: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True, msk: Tensor = None):
        embs = self.get_embs(x=x, adj=adj, diff=diff, is_sparse=is_sparse)
        return embs.cpu().to_numpy()

def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)

class SUGRL_Fast(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2):
        super(SUGRL_Fast, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.A = None
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj=None):
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, 0.2, training=self.training)
        if self.sparse:
            h_p = torch.spmm(adj, h_p_0)
        else:
            h_p = torch.mm(adj, h_p_0)
        return h_a, h_p

    def embed(self,  seq_a , adj=None ):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()
