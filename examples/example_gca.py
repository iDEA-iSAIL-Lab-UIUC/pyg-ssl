import os
import torch

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from src.methods.gca import GCA_Encoder, GRACE
from torch_geometric.datasets import WikiCS, Amazon, Coauthor
from src.augment.gca_augments import get_activation, get_base_model
from src.trainer.gca_trainer import GCATrainer

from src.evaluation import LogisticRegression
from src.utils import create_data

dataset_name = 'Amazon-Photo'
dataset = WikiCS(root='pyg_data', transform=T.NormalizeFeatures())
data_loader = DataLoader(dataset)
if dataset_name == 'Amazon-Photo':
    dataset = Amazon(root='pyg_data', name='photo', transform=T.NormalizeFeatures())
    data_loader = DataLoader(dataset)
if dataset_name == 'Coauthor-CS':
    dataset = Coauthor(root='pyg_data', name='cs', transform=T.NormalizeFeatures())
    data_loader = DataLoader(dataset)


# ---> load model
encoder = GCA_Encoder(in_channels=dataset.num_features, out_channels=256, activation=get_activation('prelu'), base_model=get_base_model('GCNConv'), k=2)
method = GRACE(encoder=encoder, loss_function=None, num_hidden=256, num_proj_hidden=32, tau=0.4)

# ---> train
trainer = GCATrainer(method=method, data_loader=data_loader, drop_scheme='degree', dataset_name='WikiCS', device="cuda:1", n_epochs=1000)
trainer.train()

# ---> evaluation
data = dataset.data.to(method.device)
embs = method.get_embs(data)
lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=10, device="cuda:1")
if dataset_name == 'WikiCS':
    data.train_mask = torch.transpose(data.train_mask, 0, 1)
    data.val_mask = torch.transpose(data.val_mask, 0, 1)
if dataset_name == 'Amazon-Photo' or dataset_name == 'Coauthor-CS':
    data = create_data.create_masks(data.cpu())
lg(embs=embs, dataset=data)