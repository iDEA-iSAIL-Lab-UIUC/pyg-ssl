from src.augment import *
from src.methods.regcl import ReGCL, ReGCLEncoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
# from src.datasets import Planetoid, Entities, Amazon, WikiCS, Coauthor
from src.evaluation import LogisticRegression
import torch
from src.config import load_yaml
from src.utils.create_data import create_masks
from src.utils.add_adj import add_adj_t
from src.data.data_non_contrast import Dataset
import argparse
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from src.evaluation import LogisticRegression


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--mode', type=int, default=4)
# parser.add_argument('--lr', type=float, default=5e-4)
# parser.add_argument('--lr2', type=float, default=1e-2)
# parser.add_argument('--tau', type=float, default=0.2)
# parser.add_argument('--dfr1', type=float, default=0.4)
# parser.add_argument('--dfr2', type=float, default=0.3)
# parser.add_argument('--der1', type=float, default=0.0)
# parser.add_argument('--der2', type=float, default=0.4)
# parser.add_argument('--lv', type=int, default=1)
# parser.add_argument('--cutway', type=int, default=2)
# parser.add_argument('--cutrate', type=float, default=1.0)
# parser.add_argument('--wd', type=float, default=0)
# parser.add_argument('--wd2', type=float, default=1e-4)
# parser.add_argument('--num_layers', type=int, default=2)
# parser.add_argument('--num_hidden', type=int, default=512)
# parser.add_argument('--num_proj_hidden', type=int, default=512)
# parser.add_argument('--test', action='store_true', default=False)
# parser.add_argument('--num_epochs', type=int, default=50)
# args = parser.parse_args()


    
    

#config.optim.patience
############################################



# load the configuration file
config = load_yaml('./configuration/graphregcl_amazon.yml')
# config = load_yaml('./configuration/graphregcl_coauthor.yml')
# config = load_yaml('./configuration/graphregcl_wikics.yml')

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

# x hape = torch.Size([2708, 1433])
#dataset = add_adj_t(dataset)
# data = dataset[0]
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

# accs = []
# for i in range(10):
#     method.eval()
#     z = method._forward(data.x, data.edge_index, 1, None, None)
#     acc = method.evaluation(z, data.y, args.dataset, device, data, learning_rate2, weight_decay2)
#     accs.append(acc)

# accs = torch.tensor(accs)
# fin_acc=torch.mean(accs)
# fin_std=torch.std(accs)
# print('fin_accuracy',fin_acc,'fin_std',fin_std)