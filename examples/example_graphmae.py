from src.methods.graphmae import GraphMAE, EncoderDecoder, load_graph_classification_dataset, setup_loss_fn, collate_fn
from src.trainer import SimpleTrainer
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import torch
import numpy as np
from src.config import load_yaml
import os

config = load_yaml('./configuration/graphmae_imdb_m.yml')
torch.manual_seed(config.torch_seed)
np.random.seed(config.torch_seed)
device = torch.device("cuda:{}".format(config.gpu_idx) if torch.cuda.is_available() and config.use_cuda else "cpu")

current_folder = os.path.abspath('')
print(current_folder)
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config.dataset.root, config.dataset.name)
# -------------------- Data --------------------
dataset, num_features = load_graph_classification_dataset(config.dataset.name, raw_dir=path)
train_idx = torch.arange(len(dataset))
train_sampler = SubsetRandomSampler(train_idx)
eval_loader = GraphDataLoader(dataset, collate_fn=collate_fn, batch_size=config.dataset.batch_size, shuffle=False)
in_channels = max(num_features, 1)

# ------------------- Method -----------------
pooling = 'mean'
if pooling == "mean":
    pooler = AvgPooling()
elif pooling == "max":
    pooler = MaxPooling()
elif pooling == "sum":
    pooler = SumPooling()
else:
    raise NotImplementedError
encoder = EncoderDecoder(GNN=config.model.encoder_type, enc_dec="encoding", in_channels=in_channels,
                         hidden_channels=config.model.hidden_channels, num_layers=config.model.encoder_layers)
decoder = EncoderDecoder(GNN=config.model.decoder_type, enc_dec="decoding", in_channels=config.model.hidden_channels,
                         hidden_channels=in_channels, num_layers=config.model.decoder_layers)
loss_function = setup_loss_fn(config.model.loss_fn, alpha_l=config.model.alpha_l)
method = GraphMAE(encoder=encoder, decoder=decoder, hidden_channels=512, argument=config, loss_function=loss_function)
method.device = device

# ------------------ Trainer --------------------
trainer = SimpleTrainer(method=method, data_loader=dataset, device=device, n_epochs=config.optim.max_epoch,
                        lr=config.optim.base_lr)
trainer.train()

# ------------------ Evaluation -------------------
method.evaluation(pooler, eval_loader)

