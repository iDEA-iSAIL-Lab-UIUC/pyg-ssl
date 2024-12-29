from torch_geometric.loader import DataLoader
from src.methods.infograph import InfoGraph, Encoder
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression
from torch_geometric.datasets import TUDataset, Entities
import os
from torch_geometric.nn import GINConv
from src.config import load_yaml
import torch
import numpy as np


config = load_yaml('./configuration/infograph_mutag.yml')
# config = load_yaml('./configuration/infograph_imdb_b.yml')
# config = load_yaml('./configuration/infograph_imdb_m.yml')
torch.manual_seed(config.torch_seed)
np.random.seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

# -------------------- Data --------------------
print(os.path.dirname(os.path.realpath(__file__)))
current_folder = os.path.abspath('')
print(current_folder)
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config.dataset.root, config.dataset.name)
if config.dataset.name in ['IMDB-B', 'IMDB-M', 'mutag', 'COLLAB', 'PROTEINS']:
    # dataset = TUDataset(path, name=config.dataset.name).shuffle()
    dataset = TUDataset(path, name=config.dataset.name)
else:
    raise NotImplementedError
# dataset.x = torch.rand(dataset.y.shape[0], 100)
data_loader = DataLoader(dataset, batch_size=config.dataset.batch_size)

in_channels = max(dataset.num_features, 1)

# ------------------- Method -----------------
#
encoder = Encoder(in_channels=in_channels, hidden_channels=config.model.hidden_channels,
                  num_layers=config.model.n_layers, GNN=GINConv)
method = InfoGraph(encoder=encoder, hidden_channels=config.model.hidden_channels, num_layers=config.model.n_layers,
                   prior=False)

# # # ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, n_epochs=config.optim.max_epoch,
                        lr=config.optim.base_lr)
trainer.train()

# ------------------ Evaluator -------------------
method.eval()
data_pyg = dataset.data.to(method.device)
y, embs = method.get_embs(data_loader)

data_pyg.x = embs
lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay,
                        max_iter=config.classifier.max_epoch, n_run=1, device=device)
lg(embs=embs, dataset=data_pyg)
