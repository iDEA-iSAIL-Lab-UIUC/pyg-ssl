from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList, AugmentorDict, RandomMaskChannel
from src.methods import BGRL, BGRLEncoder
from src.trainer import NonContrastTrainer
from torch_geometric.loader import DataLoader
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.evaluation import LogisticRegression
from src.data.data_non_contrast import Dataset
import torch, copy
import numpy as np
from src.config import load_yaml


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
config = load_yaml('./configuration/bgrl_photo.yml')
# config = load_yaml('./configuration/bgrl_cs.yml')
# config = load_yaml('./configuration/bgrl_wikics.yml')
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")
# WikiCS, cora, citeseer, pubmed, photo, computers, cs, and physics
data_name = config.dataset.name
root = config.dataset.root
# Data
# pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
# if data_name=="cora":
#     dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
# elif data_name=="photo":
#     dataset = Amazon(root="pyg_data", name="Photo", pre_transform=pre_transforms)
#     dataset.data = create_masks(dataset.data, data_name)
# elif data_name=="wikics":
#     dataset = WikiCS(root="pyg_data", pre_transform=pre_transforms, is_undirected=False)
#     dataset.data = create_masks(dataset.data, data_name)
# elif data_name=="coauthorcs":
#     dataset = Coauthor(root="pyg_data", name="CS", pre_transform=pre_transforms)
#     dataset.data = create_masks(dataset.data, data_name)

dataset = Dataset(root=root, name=data_name)
if not hasattr(dataset, "adj_t"):
    data = dataset.data
    dataset.data.adj_t = torch.sparse_coo_tensor(data.edge_index, torch.ones_like(data.edge_index[0]), [data.x.shape[0], data.x.shape[0]]).coalesce()
data_loader = DataLoader(dataset)
# dataset.data.x[7028] = torch.zeros((300))

# Augmentation
augment_1 = AugmentorList([RandomDropEdge(config.model.aug_edge_1), RandomMaskChannel(config.model.aug_mask_1)])
augment_2 = AugmentorList([RandomDropEdge(config.model.aug_edge_2), RandomMaskChannel(config.model.aug_mask_2)])
augment = AugmentorDict({"augment_1":augment_1, "augment_2":augment_2})

# ------------------- Method -----------------
if data_name=="cora":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[2048])
elif data_name=="photo":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="wikics":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[512, 256])
elif data_name=="cs":
    student_encoder = BGRLEncoder(in_channel=dataset.x.shape[1], hidden_channels=[256])

teacher_encoder = copy.deepcopy(student_encoder)

method = BGRL(student_encoder=student_encoder, teacher_encoder = teacher_encoder, data_augment=augment)

# ------------------ Trainer --------------------
trainer = NonContrastTrainer(method=method, data_loader=data_loader, device=device, use_ema=True, \
                        moving_average_decay=config.optim.moving_average_decay, lr=config.optim.base_lr, 
                        weight_decay=config.optim.weight_decay, dataset=dataset, n_epochs=config.optim.max_epoch,\
                        patience=config.optim.patience)
trainer.train()

# ------------------ Evaluator -------------------
method.eval()
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.edge_index).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=20, device=device)
lg(embs=embs, dataset=data_pyg)
