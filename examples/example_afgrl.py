from src.augment import NeighborSearch_AFGRL
from src.methods import AFGRLEncoder, AFGRL
from src.trainer import NonContrastTrainer
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, Amazon, WikiCS
from src.evaluation import LogisticRegression
import torch, copy
import torch_geometric
from src.data.data_non_contrast import Dataset
import numpy as np
from src.config import load_yaml


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
config = load_yaml('./configuration/afgrl_cs.yml')
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

# WikiCS, cora, citeseer, pubmed, photo, computers, cs, and physics
data_name = config.dataset.name
root = config.dataset.root

dataset = Dataset(root=root, name=data_name)
if not hasattr(dataset, "adj_t"):
    data = dataset.data
    dataset.data.adj_t = torch.sparse_coo_tensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]]).coalesce()
data_loader = DataLoader(dataset)
data = dataset.data

adj_ori_sparse = torch.sparse_coo_tensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]]).to(device).coalesce()

# Augmentation
augment = NeighborSearch_AFGRL(device=device, num_centroids=config.model.num_centroids, num_kmeans=config.model.num_kmeans, clus_num_iters=config.model.clus_num_iters)

# ------------------- Method -----------------
if data_name=="cora":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[2048])
elif data_name=="photo":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 512])
elif data_name=="wikics":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="cs":
    student_encoder = AFGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
teacher_encoder = copy.deepcopy(student_encoder)

method = AFGRL(student_encoder=student_encoder, teacher_encoder = teacher_encoder, data_augment=augment, adj_ori = adj_ori_sparse, topk=config.model.topk)


# ------------------ Trainer --------------------
trainer = NonContrastTrainer(method=method, data_loader=data_loader, device=device, use_ema=True, 
                        moving_average_decay=config.optim.moving_average_decay, lr=config.optim.base_lr, 
                        weight_decay=config.optim.weight_decay, n_epochs=config.optim.max_epoch, dataset=dataset,
                        patience=config.optim.patience)
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=20, device=device)
lg(embs=embs, dataset=data_pyg)