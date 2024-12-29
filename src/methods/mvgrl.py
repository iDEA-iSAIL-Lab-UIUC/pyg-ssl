from .base import BaseMethod
from torch_geometric.typing import *
from src.augment import ComputePPR, ComputeHeat, SumEmb
from typing import Optional, Callable, Union
from src.typing import AugmentType
from .utils import AvgReadout
from src.loader import AugmentDataLoader
import torch


class MVGRL(BaseMethod):
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: str="avg",
                 readout_act: Callable=torch.nn.Sigmoid(),
                 diff: AugmentType = ComputePPR(),
                 is_sparse = False,
                 sample_size: int = 1000) -> None:

        super().__init__(encoder=encoder)

        self.readout = self.readout = SumEmb(readout, readout_act)
        self.sigmoid = torch.nn.Sigmoid()
        self.is_sparse = is_sparse
        self.sample_size = sample_size
        self.corrput = diff
        self.discriminator = MVGRLDiscriminator(hidden_channels=hidden_channels)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
    
    def apply_data_augment(self, batch, batch_size = 8):
        batch, batch_neg = batch
        x_pos = batch.x
        adj = batch.adj_t.to_dense()
        x_neg = batch_neg.x
        diff = batch_neg.adj_t.to_dense()
        ft_size = x_pos.shape[1]

        lbl_1 = torch.ones(batch_size, self.sample_size * 2)
        lbl_2 = torch.zeros(batch_size, self.sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self._device)

        idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1, batch_size)
        ba = torch.zeros((batch_size, self.sample_size, self.sample_size)).to(self._device)
        bd = torch.zeros((batch_size, self.sample_size, self.sample_size)).to(self._device)
        bf = torch.zeros((batch_size, self.sample_size, ft_size)).to(self._device)
        for i in range(len(idx)):
            ba[i] = adj[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
            bd[i] = diff[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
            bf[i] = x_pos[idx[i]: idx[i] + self.sample_size]

        if self.is_sparse:
            ba = ba.to_sparse()
            bd = bd.to_sparse()

        idx = np.random.permutation(self.sample_size)
        shuf_fts = bf[:, idx, :]
        
        return bf, shuf_fts, ba, bd, lbl

    def apply_emb_augment(self, h_pos):
        return h_pos

    def get_loss(self, logits, lbl):
        loss = self.loss_func(logits, lbl)
        return loss
    
    def apply_data_augment_offline(self, dataloader):
        batch_list = []
        for i, batch in enumerate(dataloader):
            batch = batch.to(self._device)
            batch_aug = self.corrput(batch)
            batch_list.append((batch, batch_aug))
        new_loader = AugmentDataLoader(batch_list=batch_list)
        return new_loader

    def forward(self, batch):
        # 1. data augmentation
        bf, shuf_fts, ba, bd, lbl = self.apply_data_augment(batch)

        # 2. get embeddings
        embs= self.encoder(bf, shuf_fts, ba, bd, self.is_sparse)

        # 3. emb augmentation
        logits = self.discriminator(embs)

        # 4. get loss
        loss = self.get_loss(logits, lbl)
        return loss
    
    def get_embs(self, data, data_neg):
        return self.encoder(data.x, data_neg.x, data.adj_t, data_neg.adj_t, False)['final'].detach()


class MVGRLBaseEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = True):
        super(MVGRLBaseEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.act = act

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(hidden_channels))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, is_sparse=False):
        seq_fts = self.fc(seq)
        if is_sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj = adj.to_dense()
            if len(adj.shape) == 2:
                out = torch.mm(adj, seq_fts)
            else:
                out = torch.bmm(adj, seq_fts)
            
        if self.bias is not None:
            out += self.bias
        return self.act(out)
    
class MVGRLEncoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 512,
                 act: torch.nn = torch.nn.PReLU(),
                 bias: bool = True):
        super(MVGRLEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.act = act
        self.encoder_1 = MVGRLBaseEncoder(in_channels, hidden_channels, act, bias)
        self.encoder_2 = MVGRLBaseEncoder(in_channels, hidden_channels, act, bias)
        self.readout = AvgReadout()
        self.sigmoid = torch.nn.Sigmoid()      

    def forward(self, x: Tensor, x_neg: Tensor, adj: Adj, diff: Adj, is_sparse: bool = True,
                msk: Tensor = None):
        h_1 = self.encoder_1(x, adj, is_sparse)
        c_1 = self.readout(h_1)
        c_1 = self.sigmoid(c_1)
        h_2 = self.encoder_2(x, diff, is_sparse)
        c_2 = self.readout(h_2)
        c_2 = self.sigmoid(c_2)
        h_3 = self.encoder_1(x_neg, adj, is_sparse)
        h_4 = self.encoder_2(x_neg, diff, is_sparse)
        embs = {'c1':c_1, 'c2':c_2, 'h1':h_1, 'h2':h_2, 'h3':h_3, 'h4':h_4, 'final': (h_1+h_2).detach()}
        return embs


class MVGRLDiscriminator(torch.nn.Module):
    def __init__(self, hidden_channels: int = 512):
        super().__init__()
        self.f_k = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embs: dict):
        c_x1 = torch.unsqueeze(embs['c1'], 1)
        c_x1 = c_x1.expand_as(embs['h1']).contiguous()

        c_x2 = torch.unsqueeze(embs['c2'], 1)
        c_x2 = c_x2.expand_as(embs['h2']).contiguous()

        sc_1 = torch.squeeze(self.f_k(embs['h2'], c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(embs['h1'], c_x2), 2)

        sc_3 = torch.squeeze(self.f_k(embs['h4'], c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(embs['h3'], c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits
