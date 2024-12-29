import torch
import numpy as np
from torch_geometric.loader import DataLoader


from src.datasets import DBLP
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import HeCo, Sc_encoder, Mp_encoder, HeCoDBLPTransform
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression
from src.utils import create_data

import pdb


    

# -------------------- Data --------------------
dataset = DBLP(root="DBLP_data", pre_transform=HeCoDBLPTransform())
data_loader = DataLoader(dataset)
data_loader.dataset._data.cuda()

data_loader.dataset._data['nei_index'] = [i.to('cpu') for i in data_loader.dataset._data['nei_index']]

# ------------------- Method -------------------
encoder1 = Mp_encoder(P=3, hidden_dim=64, attn_drop=0.35)
encoder2 = Sc_encoder(hidden_dim=64, sample_rate=[6], nei_num=1, attn_drop=0.35)

feats = data_loader.dataset._data['feats']
feats_dim_list = [i.shape[1] for i in feats]
method = HeCo(encoder1=encoder1, encoder2=encoder2, feats_dim_list = feats_dim_list)
method.cuda()


# ------------------ Trainer -------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0", n_epochs=500, lr=0.0008, patience=30)
trainer.train()


# ----------------- Evaluator ------------------
data_pyg = dataset._data['author'].to(method.device)
embs = method.get_embs(data_loader.dataset._data['feats'], data_loader.dataset._data['mps']).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
data_pyq = create_data.create_masks(data_pyg.cpu())
lg(embs=embs, dataset=data_pyg)
