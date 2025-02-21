from src.augment import ComputePPR, ComputeHeat
from src.methods import MVGRL, MVGRLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from torch_geometric.datasets import Planetoid, Amazon, WikiCS, Coauthor
from src.evaluation import LogisticRegression
import torch
from src.config import load_yaml
from src.utils.create_data import create_masks
from src.utils.add_adj import add_adj_t

# load the configuration file
config = load_yaml('./configuration/mvgrl_cora.yml')
torch.manual_seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cuda")


# -------------------- Data --------------------
if config.dataset.name == 'cora':
    pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
    dataset = Planetoid(root='pyg_data', name='cora', pre_transform=pre_transforms)
elif config.dataset.name == 'Amazon':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = Amazon(root='pyg_data', name='Photo', pre_transform=pre_transforms)
elif config.dataset.name == 'WikiCS':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = WikiCS(root='pyg_data', pre_transform=pre_transforms)
elif config.dataset.name == 'coauthor':
    pre_transforms = NormalizeFeatures(ord=1)
    dataset = Coauthor(root='pyg_data', name='CS', pre_transform=pre_transforms)
else:
    raise 'please specify the correct dataset root'
if config.dataset.name in ['Amazon', 'WikiCS', 'coauthor']:
    dataset.data = create_masks(dataset.data, config.dataset.name)
dataset = add_adj_t(dataset)
data_loader = DataLoader(dataset, batch_size=config.model.batch_size)


# ------------------- Method -----------------
encoder = MVGRLEncoder(in_channels=config.model.in_channels, hidden_channels=config.model.hidden_channels)
method = MVGRL(encoder=encoder, diff=ComputeHeat(t = config.model.t) if config.model.aug_type == 'heat' else ComputePPR(alpha = config.model.alpha), hidden_channels=config.model.hidden_channels)


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, n_epochs=config.optim.max_epoch, patience=config.optim.patience)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
data_neg = method.corrput(data_pyg).to(method.device)
embs = method.get_embs(data_pyg, data_neg)

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=50, device=device)
lg(embs=embs, dataset=data_pyg)

