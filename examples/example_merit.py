from src.methods import GCN, Merit
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from torch_geometric.datasets import Amazon, WikiCS,Coauthor
from src.utils.create_data import create_masks
from src.evaluation import LogisticRegression
import torch 
import yaml
from src.utils.add_adj import add_adj_t
from sklearn.impute import SimpleImputer


torch.manual_seed(0)
config = yaml.safe_load(open("./configuration/merit.yml", 'r', encoding='utf-8').read())
# data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
data_name = config['dataset']

if data_name=="photo": #91.4101
    dataset = Amazon(root="pyg_data", name="photo", pre_transform=pre_transforms) 
elif data_name=="coauthor": # 92.0973
    dataset = Coauthor(root="pyg_data", name='cs', transform=pre_transforms)
elif data_name=="wikics": #82.0109
    dataset = WikiCS(root="pyg_data", transform=T.NormalizeFeatures())
    dataset = add_adj_t(dataset)
    nan_mask = torch.isnan(dataset[0].x)
    imputer = SimpleImputer()
    dataset[0].x = torch.tensor(imputer.fit_transform(dataset[0].x))

data_loader = DataLoader(dataset)
data = dataset.data


# ------------------- Method -----------------
encoder = GCN(in_ft=data.x.shape[1], out_ft=512, projection_hidden_size=config["projection_hidden_size"],
                  projection_size=config["projection_size"])
method = Merit(encoder=encoder, data = data, config=config,device="cuda:0",is_sparse=True)

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg, data_pyg.adj_t).detach()

lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=2000, n_run=50, device="cuda")
create_masks(data=data_pyg.cpu())
lg(embs=embs, dataset=data_pyg)
