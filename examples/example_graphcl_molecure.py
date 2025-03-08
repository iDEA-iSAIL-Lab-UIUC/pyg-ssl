from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList
from src.methods.infograph import InfoGraph, Encoder
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from src.evaluation import LogisticRegression
from src.config import load_yaml
from torch_geometric.nn import GINConv
import torch
import os
import numpy as np


torch.manual_seed(0)
config = load_yaml('./configuration/infograph_mutag.yml')
torch.manual_seed(config.torch_seed)
np.random.seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

# data
# -------------------- Data --------------------
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config.dataset.root)
if config.dataset.name in ['IMDB-B', 'IMDB-M', 'mutag', 'COLLAB', 'PROTEINS']:
    dataset = TUDataset(path, name=config.dataset.name.upper())
else:
    raise NotImplementedError
data_loader = DataLoader(dataset, batch_size=config.dataset.batch_size)

in_channels = max(dataset.num_features, 1)

### to handle molecular graph, graphcl use infograph as the backbone. ###
# ------------------- Method -----------------
#
encoder = Encoder(in_channels=in_channels, hidden_channels=config.model.hidden_channels,
                  num_layers=config.model.n_layers, GNN=GINConv)
method = InfoGraph(encoder=encoder, hidden_channels=config.model.hidden_channels, num_layers=config.model.n_layers,
                   prior=False)

# Augmentation
aug_type = 'mask'
if aug_type == 'edge':
    augment_neg = RandomDropEdge()
elif aug_type == 'mask':
    augment_neg = RandomMask()
elif aug_type == 'node':
    augment_neg = RandomDropNode()
elif aug_type == 'subgraph':
    augment_neg = AugmentSubgraph()
else:
    assert False
augment_neg = AugmentorList([RandomDropEdge(), RandomMask()])


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()


# ------------------ Evaluator -------------------
method.eval()
data_pyg = dataset.data.to(method.device)
y, embs = method.get_embs(data_loader)

data_pyg.x = embs
lg = LogisticRegression(lr=config.classifier.base_lr, weight_decay=config.classifier.weight_decay,
                        max_iter=config.classifier.max_epoch, n_run=1, device=device)
lg(embs=embs, dataset=data_pyg)






