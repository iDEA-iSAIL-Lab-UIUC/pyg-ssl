from src.augment import *
from src.methods.regcl import ReGCL, ReGCLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures
from src.evaluation import LogisticRegression
import torch
from src.config import load_yaml
from src.utils.create_data import create_masks
from src.utils.add_adj import add_adj_t
from src.data.data_non_contrast import Dataset
import argparse
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from src.evaluation import LogisticRegression

# load the configuration file
config = load_yaml('./configuration/graphregcl_amazon.yml')

# assert config.gpu_idx in range(0, 8)
torch.cuda.set_device(config.gpu_idx)

learning_rate = config.optim.lr
learning_rate2 = config.optim.lr2
drop_edge_rate_1 = config.optim.der1
drop_edge_rate_2 = config.optim.der2
drop_feature_rate_1 = config.optim.dfr1
drop_feature_rate_2 = config.optim.dfr2
tau = config.optim.tau
mode = config.model.mode
nei_lv = config.optim.lv
cutway = config.optim.cutway
cutrate = config.optim.cutrate
num_hidden = config.model.num_hidden
num_proj_hidden = config.model.num_proj_hidden
activation = F.relu
base_model = GCNConv
num_layers = config.model.num_layers
num_epochs = config.optim.max_epoch
weight_decay = config.optim.wd
weight_decay2 = config.optim.wd2

torch.manual_seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

data_name = config.dataset.name
root = config.dataset.root


dataset = Dataset(root=root, name=data_name, transform=NormalizeFeatures())

data = dataset.data.to(device)
data_loader = DataLoader(dataset, )

#
# ------------------- Method -----------------
encoder = ReGCLEncoder(dataset.num_features, num_hidden, activation, mode, base_model=base_model, k=num_layers, cutway=cutway, cutrate=cutrate, tau=tau).to(device)
method = ReGCL(config, encoder, num_hidden, num_proj_hidden, mode, tau).to(device)


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader,
                        device=device, 
                        n_epochs=config.optim.max_epoch,
                        patience=config.optim.patience)
trainer.train()


# ------------------ Evaluator -------------------
method.eval()
data_pyg = dataset.data.to(method.device)
y, embs = method.get_embs(data)

lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay,
                        max_iter=config.classifier.max_epoch,
                        n_run=1, device=device)
lg(embs=embs, dataset=data_pyg)