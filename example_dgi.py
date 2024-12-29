from torch_geometric.loader import DataLoader
from src.datasets import Planetoid
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.methods import DGI, DGIEncoder
from src.trainer import SimpleTrainer

from src.evaluation import LogisticRegression


# -------------------- Data --------------------
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
data_loader = DataLoader(dataset)


encoder = DGIEncoder(in_channels=1433, hidden_channels=512)
method = DGI(encoder=encoder, hidden_channels=512)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()

# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg)

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device="cuda")
lg(embs=embs, dataset=data_pyg)
