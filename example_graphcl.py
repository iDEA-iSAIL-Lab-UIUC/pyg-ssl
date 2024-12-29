from src.augment import *
from src.methods import GraphCL, GraphCLEncoder
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

# load the configuration file
# config = load_yaml('./configuration/graphcl_amazon.yml')
# config = load_yaml('./configuration/graphcl_coauthor.yml')
config = load_yaml('./configuration/graphcl_wikics.yml')
# config = load_yaml('./configuration/graphcl_cora.yml')
torch.manual_seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

# data
# if config.dataset.name == 'pyg_data':
#     pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
#     # dataset = Planetoid(root=config.dataset.root, name=config.dataset.name, pre_transform=pre_transforms)
#     dataset = Planetoid(root='pyg_data', name=config.dataset.name)
# elif config.dataset.name == 'Amazon':
#     pre_transforms = NormalizeFeatures(ord=1)
#     dataset = Amazon(root='pyg_data', name='Photo', pre_transform=pre_transforms)
# elif config.dataset.name == 'WikiCS':
#     pre_transforms = NormalizeFeatures(ord=1)
#     dataset = WikiCS(root='pyg_data', pre_transform=pre_transforms)
# elif config.dataset.name == 'coauthor':
#     pre_transforms = NormalizeFeatures(ord=1)
#     dataset = Coauthor(root='pyg_data', name='CS', pre_transform=pre_transforms)
# else:
#     raise 'please specify the correct dataset root'

data_name = config.dataset.name
root = config.dataset.root
dataset = Dataset(root=root, name=data_name)
if config.dataset.name in ['Amazon', 'WikiCS', 'coauthor']:
    dataset.data = create_masks(dataset.data, config.dataset.name)
dataset = add_adj_t(dataset)
data_loader = DataLoader(dataset)

# Augmentation
aug_type = config.model.aug_type
if aug_type == 'edge':
    augment_neg = AugmentorList([RandomDropEdge()])
elif aug_type == 'mask':
    augment_neg = AugmentorList([RandomMask()])
elif aug_type == 'node':
    augment_neg = AugmentorList([RandomDropNode()])
elif aug_type == 'subgraph':
    augment_neg = AugmentorList([AugmentSubgraph()])
else:
    assert 'unrecognized augmentation method'
#
# ------------------- Method -----------------
encoder = GraphCLEncoder(in_channels=config.model.in_channels, hidden_channels=config.model.hidden_channels)
method = GraphCL(encoder=encoder, corruption=augment_neg, hidden_channels=config.model.hidden_channels)
method.augment_type = aug_type


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader,
                        device=device, n_epochs=config.optim.max_epoch,
                        patience=config.optim.patience)
trainer.train()


# ------------------ Evaluator -------------------
method.eval()
# renew dataset to get back train_mask and value_mask from .T
dataset = Dataset(root=root, name=data_name)
dataset = add_adj_t(dataset)
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg)
lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay,
                        max_iter=config.classifier.max_epoch, n_run=1, device=device)
lg(embs=embs, dataset=data_pyg)






