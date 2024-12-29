import os.path as osp
from torch_geometric.loader import DataLoader


from src.datasets import DBLP, aminer, ACM, FreebaseMovies
from src.methods import HeCo, Sc_encoder, Mp_encoder, HeCoDBLPTransform
from src.trainer import SimpleTrainer
from src.evaluation import LogisticRegression
from src.utils import create_data

# pre-tuned parameters
params = {}
params['acm'] = {}
params['acm']['P'] = 2
params['acm']['sample_rate'] = [7, 1]
params['acm']['nei_num'] = 2
params['acm']['target_type'] = 'paper'
params['acm']['patience'] = 5
params['acm']['tau'] = 0.8
params['acm']['feat_drop'] = 0.3
params['acm']['attn_drop'] = 0.5
params['acm']['eva_lr'] = 0.07

params['dblp'] = {}
params['dblp']['P'] = 3
params['dblp']['sample_rate'] = [6]
params['dblp']['nei_num'] = 1
params['dblp']['target_type'] = 'author'
params['dblp']['patience'] = 30
params['dblp']['tau'] = 0.9
params['dblp']['feat_drop'] = 0.4
params['dblp']['attn_drop'] = 0.35
params['dblp']['eva_lr'] = 0.01
    
params['freebase_movies'] = {}
params['freebase_movies']['P'] = 3
params['freebase_movies']['sample_rate'] = [1, 18, 2]
params['freebase_movies']['nei_num'] = 3
params['freebase_movies']['target_type'] = 'movie'
params['freebase_movies']['patience'] = 20
params['freebase_movies']['tau'] = 0.5
params['freebase_movies']['feat_drop'] = 0.1
params['freebase_movies']['attn_drop'] = 0.3
params['freebase_movies']['eva_lr'] = 0.01

params['aminer'] = {}
params['aminer']['P'] = 2
params['aminer']['sample_rate'] = [3, 8]
params['aminer']['nei_num'] = 2
params['aminer']['target_type'] = 'paper'
params['aminer']['patience'] = 40
params['aminer']['tau'] = 0.5
params['aminer']['feat_drop'] = 0.5
params['aminer']['attn_drop'] = 0.5
params['aminer']['eva_lr'] = 0.01


# -------------------- Data --------------------
# dataset = DBLP(root="DBLP_data", pre_transform=HeCoDBLPTransform())
dataset_name = 'acm'
if dataset_name == 'acm':
    dataset = ACM(root=osp.join("./datasets", dataset_name))
elif dataset_name == 'dblp':
    dataset = DBLP(root=osp.join("./datasets", dataset_name), pre_transform=HeCoDBLPTransform())
elif dataset_name == 'freebase_movies':
    dataset = FreebaseMovies(root=osp.join("./datasets", dataset_name))
elif dataset_name == 'aminer':
    dataset = aminer(root=osp.join("./datasets", dataset_name))

data_loader = DataLoader(dataset)


# ------------------- Method -------------------
encoder1 = Mp_encoder(P=params[dataset_name]['P'], hidden_dim=64, attn_drop=params[dataset_name]['attn_drop'])
encoder2 = Sc_encoder(hidden_dim=64, sample_rate=params[dataset_name]['sample_rate'], nei_num=params[dataset_name]['nei_num'], attn_drop=params[dataset_name]['attn_drop'])

feats = data_loader.dataset._data['feats']
feats_dim_list = [i.shape[1] for i in feats]
method = HeCo(encoder1=encoder1, encoder2=encoder2, feats_dim_list = feats_dim_list, feat_drop=params[dataset_name]['feat_drop'], tau=params[dataset_name]['tau'])
method.cuda()


# ------------------ Trainer -------------------
trainer = SimpleTrainer(method=method, data_loader=data_loader, device="cuda:0", n_epochs=500, lr=0.0008, patience=params[dataset_name]['patience'])
trainer.train()


# ----------------- Evaluator ------------------
data_pyg = dataset._data[params[dataset_name]['target_type']].to(method.device)
embs = method.get_embs(data_loader.dataset._data['feats'], data_loader.dataset._data['mps']).detach()

lg = LogisticRegression(lr=params[dataset_name]['eva_lr'], weight_decay=0, max_iter=100, n_run=50, device="cuda")
data_pyq = create_data.create_masks(data_pyg.cpu())
lg(embs=embs, dataset=data_pyg)
