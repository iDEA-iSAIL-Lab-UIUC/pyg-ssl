import torch
import torch.nn as nn

from .base import BaseMethod
from src.augment import ShuffleNode, SumEmb
from src.losses import NegativeMI

from typing import Callable


class HDMI(BaseMethod):
    def __init__(self, 
                 encoder: torch.nn.Module, 
                 in_channels: int, 
                 hidden_channels: int, 
                 n_layers: int,
                 w_extrinsic: float=1.0,
                 w_intrinsic: float=1.0,
                 w_joint: float=1.0,
                 readout: str="avg",
                 readout_act: Callable=torch.nn.Sigmoid(),
                 proj_act: Callable=torch.nn.Sigmoid(),
                 same_discriminator: bool=True):
        r"""The full model to train the encoder.
            Args:
                encoder (torch.nn.Module): the encoder to be trained.
                in_channels (int): the number of input features.
                hidden_channels (int): the channel of the hidden dimension.
                n_layers (int): the number of layers of the heterogeneous multiplex graph.
                readout (str): "avg" or "sum". Sum emebddings.
                readout_act (Callable): the activation function for readout.
                proj_act (Callable): the activation function for proj embeddings.
                same_discriminator (bool): use the same discriminator for local (layers/views) and the final emebeddings or not.
                discriminator (torch.nn.Module): the discriminator for contrastive learning.
        """
        super().__init__(encoder=encoder)
        # augments
        self.corrupt = ShuffleNode()
        self.readout = SumEmb(readout, readout_act)
        
        # losses
        self.loss_layer_e, self.loss_layer_i, self._loss_layer_j = self._init_loss(in_channels, hidden_channels)
        self.s_layer_proj, self.f_layer_proj, self.sf_layer_proj = self._init_proj(in_channels, hidden_channels, proj_act)

        if same_discriminator:
            self.loss_final_e, self.loss_final_i, self._loss_final_j  = self.loss_layer_e, self.loss_layer_i, self._loss_layer_j
            self.s_final_proj, self.f_final_proj, self.sf_final_proj = self.s_layer_proj, self.f_layer_proj, self.sf_layer_proj
        else:
            self.loss_final_e, self.loss_final_i, self._loss_final_j = self._init_loss(in_channels, hidden_channels)
            self.s_final_proj, self.f_final_proj, self.sf_final_proj = self._init_proj(in_channels, hidden_channels, proj_act)

        # weights
        self.w_e = w_extrinsic
        self.w_i = w_intrinsic
        self.w_j = w_joint

        # others
        self.n_layers = n_layers

    def _init_loss(self, in_channels, hidden_channels):
        loss_e = NegativeMI(sim_func=nn.Bilinear(hidden_channels, hidden_channels, 1))
        loss_i = NegativeMI(sim_func=nn.Bilinear(in_channels, hidden_channels, 1))
        loss_j = NegativeMI(sim_func=nn.Bilinear(hidden_channels, hidden_channels, 1))
        return loss_e, loss_i, loss_j
    
    def _init_proj(in_channels, hidden_channels, proj_act):
        s_proj = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels), proj_act])
        f_proj = nn.ModuleList([nn.Linear(in_channels, hidden_channels), proj_act])
        sf_proj = nn.ModuleList([nn.Linear(2*hidden_channels, hidden_channels), proj_act])
        return s_proj, f_proj, sf_proj

    def forward(self, feats, adj_list, sparse=False):
        """
        Args:
            features: node features or attributes. Shape: [n_nodes, in_channels].
            adj_list: a list of adjs. Each adj has the shape [N, N].
            sparse: use sparse matrix multiplcation or not.
        """
        assert self.n_layers == len(adj_list), f"The specificed number of layers is {self.n_layers}, but the input has {len(adj_list)} layers."

        # 1. data augment: get negative samples
        feats_neg = self.apply_data_augment(feats)

        # 2. get embeddings
        emb_dict_pos = self.encoder(feats, adj_list, sparse)
        emb_dict_neg = self.encoder(feats_neg, adj_list, sparse)

        # 3. emb augment: get summary embs 
        sum_dict_pos = self.apply_emb_augment(emb_dict_pos)

        # 4. get loss based on the pos & neg embs
        loss = self.get_loss(emb_dict_pos, emb_dict_neg, sum_dict_pos, feats, feats_neg)
        return loss

    def apply_data_augment(self, feats):
        feats_neg = self.corrupt(feats).to(self.feats.device)
        return feats_neg

    def apply_emb_augment(self, emb_dict):
        sum_embs_dict = {
            "final": self.readout(emb_dict["final"], dim=-2),   # [hidden_channels]
            "layers": self.readout(emb_dict["layers"], dim=-2)  # [n_layers, hidden_channels]
        }
        return sum_embs_dict
        

    def get_loss(self, emb_dict_pos, emb_dict_neg, sum_dict_pos, f_pos, f_neg):        
        # layers
        loss_layer_e = loss_layer_i = loss_layer_j = 0
        for n in range(self.n_layers):
            h_pos, h_neg = emb_dict_pos["layers"][n], emb_dict_neg["layers"][n]
            s_pos = torch.unsqueeze(sum_dict_pos["layers"][n], 0)
            loss_layer_e += self.loss_layer_e(s_pos, h_pos, s_pos, h_neg)
            loss_layer_i += self.loss_layer_i(f_pos, h_pos, f_pos, h_neg)
            loss_layer_j += self.loss_layer_j(s_pos, h_pos, f_pos, f_neg)
        loss_layer = self.w_e*loss_layer_e + self.w_i*loss_layer_i + self.w_j*loss_layer_j

        # final
        h_pos, h_neg = emb_dict_pos["final"], emb_dict_neg["final"]
        s_pos = emb_dict_pos["final"]
        loss_final_e = self.loss_final_e(s_pos, h_pos, s_pos, h_neg)
        loss_final_i = self.loss_final_i(f_pos, h_pos, f_pos, h_neg)
        loss_final_j = self.loss_final_j(s_pos, h_pos, f_pos, f_neg)
        loss_final = self.w_e*loss_final_e + self.w_i*loss_final_i + self.w_j*loss_final_j

        loss = loss_layer + loss_final
        return loss

    def loss_layer_j(self, s_pos, h_pos, f_pos, f_neg):
        loss = self._get_joint_loss(s_pos, h_pos, f_pos, f_neg, self.s_layer_proj, self.sf_layer_proj, self._loss_layer_j)
        return loss

    def loss_final_j(self, s_pos, h_pos, f_pos, f_neg):
        loss = self._get_joint_loss(s_pos, h_pos, f_pos, f_neg, self.s_final_proj, self.sf_final_proj, self._loss_final_j)
        return loss

    def _get_joint_loss(self, s_pos, h_pos, f_pos, f_neg, s_proj_fn, f_proj_fn, sf_proj_fn, loss_fn):
        s_pos = s_proj_fn(s_pos)
        f_pos, f_neg = f_proj_fn(f_pos), f_proj_fn(f_neg)
        sf_pos, sf_neg = torch.cat([s_pos, f_pos], dim=-1), torch.cat([s_pos, f_neg], dim=-1)
        sf_pos, sf_neg = sf_proj_fn(sf_pos), sf_proj_fn(sf_neg)
        loss_j = loss_fn(x=h_pos, y=sf_pos, x_ind=h_pos, y_ind=sf_neg)
        return loss_j

class HDMIEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 n_layers: int):
        """
        Args:
            in_channels: the dimension of the input features.
            hidden_channels: the dimension of the embeddings.
            n_layers: the number of meta-paths.
        """
        super(HDMIEncoder, self).__init__()
        self.gcn_list = nn.ModuleList([GCN(in_channels, hidden_channels) for _ in range(n_layers)])
        self.w_list = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(n_layers)])
        self.y_list = nn.ModuleList([nn.Linear(hidden_channels, 1) for _ in range(n_layers)])

        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)

        # for m in self.modules():
        #     self.weights_init(m)
    # def weights_init(self, m):
        # if isinstance(m, nn.Linear):
        #     torch.nn.init.xavier_uniform_(m.weight.data)
        #     if m.bias is not None:
        #         m.bias.data.fill_(0.0)

    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, features, adj_list, sparse=True):
        """
        Args:
            features: node features or attributes. Shape: [n_nodes, in_channels].
            adj_list: a list of adjs. Each adj has the shape [n_nodes, n_nodes].
            sparse: use sparse matrix multiplcation or not.
        
        Returns:
            emb_dict: a dictionary of embeddings.
            {"final": the final embedding. Shape: [dim]
             "layers": the emebeddings of different layers/views. Shape: [n_layers, n_nodes, dim]}
        """
        emb_list = []
        for i, adj in enumerate(adj_list):
            emb = torch.squeeze(self.gcn_list[i](features, adj, sparse))
            emb_list.append(emb)
        emb = self.combine_att(emb_list)
        emb_dict = {"final": emb, "layers": torch.stack(emb_list)}
        return emb_dict


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, isBias=True):
        super(GCN, self).__init__()

        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.ReLU()

        if isBias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq = self.fc(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias

        return self.act(seq)
