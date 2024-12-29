import torch
from torch_geometric.loader import DataLoader
# from torch_geometric.nn.models import GCN

from src.datasets import Planetoid
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import DGI, DGIEncoder
from src.trainer import SimpleTrainer
from src.augment import ShuffleNode

from src.evaluation import LogisticRegression, SVCRegression, RandomForestClassifier, TSNEVisulization, SimSearch, K_Means



# -------------------- Data --------------------
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)


# ------------------- Method -----------------
# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels=512, num_layers=1, act=torch.nn.PReLU()):
#         super().__init__()
#         self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, act=act)
#         self.act = act

#     def forward(self, batch):
#         edge_weight = batch.edge_weight if "edge_weight" in batch else None
#         return self.act(self.gcn(x=batch.x, edge_index=batch.edge_index, edge_weight=edge_weight))

encoder = DGIEncoder(in_channels=1433, hidden_channels=512)
method = DGI(encoder=encoder, hidden_channels=512)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()

# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = encoder(data_pyg).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)

# plot_embedding2D(embs=embs, dataset=data_pyg)
