import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
import random
import scipy.sparse as sp
from .base import BaseMethod
from torch_geometric.typing import *





class MLP(nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(h1, h2):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def contrastive_loss_wo_cross_network(h1, h2, z):
    f = lambda x: torch.exp(x)
    intra_sim = f(sim(h1, h1))
    inter_sim = f(sim(h1, h2))
    return -torch.log(inter_sim.diag() /
                     (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


def contrastive_loss_wo_cross_view(h1, h2, z):
    f = lambda x: torch.exp(x)
    cross_sim = f(sim(h1, z))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))


class Merit(BaseMethod):
    
    def __init__(self, 
                 encoder: torch.nn.Module,
                 data,
                 config,
                 device,
                 is_sparse = False,
                 loss_function: Optional[torch.nn.Module] = None):
        
        super().__init__(encoder=encoder, loss_function=loss_function)

        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(config["momentum"])
        self.online_predictor = MLP(config["projection_size"], config["prediction_size"], config["prediction_hidden_size"])
        self.beta = config["beta"]
        self.drop_edge_rate_1 = config["drop_edge"]
        self.drop_feature_rate_1 = config["drop_feat1"]
        self.drop_feature_rate_2 = config["drop_feat2"]
        self.is_sparse = is_sparse
        self.device = device
        self.config = config
        self.sample_size = config["sample_size"]
        self.data = data
                   
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, batch, batch_size = 8):
        x_pos = self.data.x
        # print(batch.adj_t)
        adj = self.data.adj_t.to_dense()
        diff = gdc(sp.csr_matrix(np.matrix(adj)), alpha=self.config["alpha"], eps=0.0001)
        diff =  torch.from_numpy(diff)
        # diff = batch_neg.adj_t.to_dense()
        ft_size = x_pos.shape[1]

        
        idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1)
        ba = adj[idx: idx + self.sample_size, idx: idx + self.sample_size]
        bd = diff[idx: idx + self.sample_size, idx: idx + self.sample_size]
        bd = sp.csr_matrix(np.matrix(bd))
        features = x_pos.squeeze(0)
        bf = features[idx: idx + self.sample_size]
        ba = sp.csr_matrix(np.matrix(ba))
        # bf = sp.csr_matrix(np.matrix(bf))

        # lbl_1 = torch.ones(batch_size, self.sample_size * 2)
        # lbl_2 = torch.zeros(batch_size, self.sample_size * 2)
        # lbl = torch.cat((lbl_1, lbl_2), 1)

        # idx = np.random.randint(0, adj.shape[-1] - self.sample_size + 1, batch_size)
        # ba = torch.zeros((batch_size, self.sample_size, self.sample_size))
        # bd = torch.zeros((batch_size, self.sample_size, self.sample_size))
        # bf = torch.zeros((batch_size, self.sample_size, ft_size))
        # for i in range(len(idx)):
        #     ba[i] = adj[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
        #     bd[i] = diff[idx[i]: idx[i] + self.sample_size, idx[i]: idx[i] + self.sample_size]
        #     bf[i] = x_pos[idx[i]: idx[i] + self.sample_size]

        # if self.is_sparse:
        #     ba = ba.to_sparse()
        #     bd = bd.to_sparse()

        aug_adj1 = aug_random_edge(ba, drop_percent=self.drop_edge_rate_1)
        aug_adj2 = bd
        aug_features1 = aug_feature_dropout(bf, drop_percent=self.drop_feature_rate_1)
        aug_features2 = aug_feature_dropout(bf, drop_percent=self.drop_feature_rate_2)

        aug_adj1 = normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
        aug_adj2 = normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

        if self.is_sparse:
            adj_1 = sparse_mx_to_torch_sparse_tensor(aug_adj1).to(self.device)
            adj_2 = sparse_mx_to_torch_sparse_tensor(aug_adj2).to(self.device)
        else:
            aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
            aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
            adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(self.device)
            adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(self.device)

        aug_features1 = aug_features1.to(self.device)
        aug_features2 = aug_features2.to(self.device)

        _, online_proj_one = self.online_encoder(aug_features1, adj_1, self.is_sparse)
        _, online_proj_two = self.online_encoder(aug_features2, adj_2, self.is_sparse)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
                      
        with torch.no_grad():
            _, target_proj_one = self.target_encoder(aug_features1,adj_1,  self.is_sparse)
            _, target_proj_two = self.target_encoder(aug_features2,adj_2,  self.is_sparse)
                       
        l1 = self.beta * contrastive_loss_wo_cross_network(online_pred_one, online_pred_two, target_proj_two.detach()) + \
            (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_one, online_pred_two, target_proj_two.detach())
        
        l2 = self.beta * contrastive_loss_wo_cross_network(online_pred_two, online_pred_one, target_proj_one.detach()) + \
            (1.0 - self.beta) * contrastive_loss_wo_cross_view(online_pred_two, online_pred_one, target_proj_one.detach())
        
        loss = 0.5 * (l1 + l2)
            
        return loss.mean()
    
    def get_embs(self, data: Tensor, adj: Adj, is_sparse: bool = True):
        embs,_ = self.online_encoder(data.x,adj.to_torch_sparse_coo_tensor(), is_sparse)
        # h_p = self.encoder_2(x, adj.to_torch_sparse_coo_tensor(), is_sparse)
        
        return embs
    

class GCN(nn.Module):
    
    def __init__(self, in_ft, out_ft, projection_hidden_size,
                  projection_size, act='prelu', bias=True):
        
        super(GCN, self).__init__()
        
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()
        self.projector = MLP(512, projection_size, projection_hidden_size)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        representations = self.act(out)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)

        return representations, projections

def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent = 0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    num_drop = int(len(row_idx)*percent)
    
    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    drop_idx = random.sample(edge_index, k = num_drop)
    
    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))
    
    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()
    
    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape = input_adj.shape)
 
    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))
    add_idx = random.sample(no_edges_cells, num_drop)    
    new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))         
    row_idx = new_row_idx + new_row_idx_1
    col_idx = new_col_idx + new_col_idx_1
    data = np.ones(len(row_idx)).tolist()
    
    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape = input_adj.shape)
        
    return new_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break
  
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_feature_dropout(input_feat, drop_percent = 0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


def aug_feature_dropout_cell(input_feat, drop_percent = 0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    input_feat_dim = aug_input_feat.shape[1]
    num_of_nodes = aug_input_feat.shape[0]   
    drop_feat_num = int(num_of_nodes * input_feat_dim * drop_percent)
    
    position = []
    number_list = [j for j in range(input_feat_dim)]
    for i in range(num_of_nodes):
      number_i = [i for k in range(input_feat_dim)]
      position += list(zip(number_i, number_list))
      
    drop_idx = random.sample(position, drop_feat_num)
    for i in range(len(drop_idx)):
        aug_input_feat[(drop_idx[i][0],drop_idx[i][1])] = 0.0
    
    return aug_input_feat

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out

def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    # print(A)
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S