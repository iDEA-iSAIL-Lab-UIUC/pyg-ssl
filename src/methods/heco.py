import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from .base import BaseMethod

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from .utils import AvgReadout
from typing import Optional, Callable, Union

@functional_transform('heco_transform_ssl')
class HeCoDBLPTransform(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, data):
        # load feature matrices
        batch = data
        type_num = [4057, 14328, 7723, 20]
        feat_a = torch.FloatTensor(preprocess_features(sp.csr_matrix(batch['author'].x.to('cpu').numpy())))
        feat_p = sp.eye(type_num[1])
        feat_p = torch.FloatTensor(preprocess_features(feat_p))
        feats = [feat_a.cuda(), feat_p.cuda()]
        data['feats'] = feats

        # generate meta paths
        pa = batch['paper', 'to', 'author'].edge_index.to('cpu').numpy().T
        pc = batch['paper', 'to', 'conference'].edge_index.to('cpu').numpy().T
        pt = batch['paper', 'to', 'term'].edge_index.to('cpu').numpy().T

        A = type_num[0]
        P = type_num[1]
        T = type_num[2]
        C = type_num[3]

        pa_ = sp.coo_matrix((np.ones(pa.shape[0]),(pa[:,0], pa[:, 1])),shape=(P,A)).toarray()
        pc_ = sp.coo_matrix((np.ones(pc.shape[0]),(pc[:,0], pc[:, 1])),shape=(P,C)).toarray()
        pt_ = sp.coo_matrix((np.ones(pt.shape[0]),(pt[:,0], pt[:, 1])),shape=(P,T)).toarray()

        apa = np.matmul(pa_.T, pa_) > 0
        apa = sp.coo_matrix(apa)

        apc = np.matmul(pa_.T, pc_) > 0
        apcpa = np.matmul(apc, apc.T) > 0
        apcpa = sp.coo_matrix(apcpa)

        apt = np.matmul(pa_.T, pt_) > 0
        aptpa = np.matmul(apt, apt.T) > 0
        aptpa = sp.coo_matrix(aptpa)

        apa_mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
        apcpa_mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
        aptpa_mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))

        mps = [apa_mp.cuda(), apcpa_mp.cuda(), aptpa_mp.cuda()]
        data['mps'] = mps

        # generate positive set
        pos_num = 1000
        p = A
        apa = apa/apa.sum(axis=-1).reshape(-1, 1)
        apcpa = apcpa/apcpa.sum(axis=-1).reshape(-1, 1)
        aptpa = aptpa/aptpa.sum(axis=-1).reshape(-1, 1)
        all = (apa + apcpa + aptpa).A.astype("float32")

        pos = np.zeros((p,p))
        k=0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k+=1
            else:
                pos[i, one] = 1
        pos = sp.coo_matrix(pos)
        pos = sparse_mx_to_torch_sparse_tensor(pos)
        data['pos'] = pos.cuda()

        # generate neighboring information
        pa2 = batch['paper', 'to', 'author'].edge_index.to('cpu').numpy().T
        a_n = {}
        for i in pa2:
            if i[1] not in a_n:
                a_n[int(i[1])]=[]
                a_n[int(i[1])].append(int(i[0]))
            else:
                a_n[int(i[1])].append(int(i[0]))
            
        keys =  sorted(a_n.keys())
        a_n = [a_n[i] for i in keys]
        a_m = []
        for i in a_n:
            a_m.append(np.array(i, dtype=np.int64))
        nei_index = np.array(a_m, dtype=object)
        nei_index = [torch.LongTensor(i) for i in nei_index]
        
        data['nei_index'] = [nei_index]

        return data


# Network Schema View Guided Encoder
class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_sc = 0
        for i in range(len(embeds)):
            z_sc += embeds[i] * beta[i]
        return z_sc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()
            one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
            embeds.append(one_type_emb)
        z_sc = self.inter(embeds)
        return z_sc


# Meta-path View Guided Encoder
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp


class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)

    def forward(self, h, mps):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        z_mp = self.att(embeds)
        return z_mp


# Contrast Loss Function
class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc


# The HeCo Algorithm
class HeCo(BaseMethod):

    def __init__(self,
                 encoder1: torch.nn.Module,
                 encoder2: torch.nn.Module,
                 feats_dim_list,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 loss_function: Optional[torch.nn.Module] = None,
                 data_argument: None = None,
                 hidden_channels: int = 64,
                 feat_drop: float = 0.3,
                 tau: float = 0.9,
                 lam: float = 0.5,
                 ) -> None:
        loss_function = loss_function if loss_function else Contrast(hidden_channels, tau, lam)
        super().__init__(encoder=encoder1, data_augment=data_argument, loss_function=loss_function)
        self.hidden_channels = hidden_channels
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.hidden_channels, bias=True)
                                      for feats_dim in feats_dim_list])

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = encoder1
        self.sc = encoder2

    def forward(self, batch):
        # TODO: DBLP-specific to heterogeneous datasets
        feats = batch['feats']
        mps = batch['mps']
        pos = batch['pos']
        nei_index = batch['nei_index'][0]
        # compute loss
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))).to(self.device))
        
        z_mp = self.mp(h_all[0], mps)
        z_sc = self.sc(h_all, nei_index)
        loss = self.loss_function(z_mp, z_sc, pos)
        return loss

    def get_embs(self, feats, mps):
        for i in range(len(mps)):
            mps[i] = mps[i].to(self.device)
        z_mp = F.elu(self.fc_list[0](feats[0])).to(self.device)
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()


# Helper Functions
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)