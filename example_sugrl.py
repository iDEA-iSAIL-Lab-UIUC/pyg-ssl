from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList
from src.methods import SugrlMLP, SugrlGCN
from src.methods import SUGRL
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Amazon, WikiCS,Coauthor
from src.utils.create_data import create_masks
from src.evaluation import LogisticRegression
import torch 
import yaml
from src.utils.add_adj import add_adj_t
from sklearn.impute import SimpleImputer

torch.manual_seed(0)

pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])

# load the configuration file
# config = yaml.safe_load(open('./configuration/sugrl_wikics.yml', 'r', encoding='utf-8').read())
config = yaml.safe_load(open("./configuration/sugrl_amazon.yml", 'r', encoding='utf-8').read())
# config = yaml.safe_load(open("./configuration/sugrl_coauthor.yml", 'r', encoding='utf-8').read())
# config = yaml.safe_load(open("./configuration/sugrl_cora.yml", 'r', encoding='utf-8').read())
print(config)
data_name = config['dataset']

if data_name=="cora":
    dataset = Planetoid(root="pyg_data", name="cora", pre_transform=pre_transforms)
if data_name=="photo": #92.9267
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

# has_nan = torch.isnan(data.x).any()

# if has_nan:
#     print("The tensor contains NaN values.")
# else:
#     print("The tensor does not contain NaN values.")


# ------------------- Method -----------------
encoder_1 = SugrlMLP(in_channels=data.x.shape[1])
encoder_2 = SugrlGCN(in_channels=data.x.shape[1])
method = SUGRL(encoder=[encoder_1,encoder_2],data = data, config=config,device="cuda:0")


# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0")
# trainer = SUGRL(model=model, data_loader=data_loader,data=dataset[0], device="cuda:0")
trainer.train()


# ------------------ Evaluator -------------------
data_pyg = dataset.data.to(method.device)
embs = method.get_embs(data_pyg.x, data_pyg.adj_t).detach()

lg = LogisticRegression(lr=0.001, weight_decay=0, max_iter=3000, n_run=10, device="cuda")
create_masks(data=data_pyg.cpu())
lg(embs=embs, dataset=data_pyg)