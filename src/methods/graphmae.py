from itertools import chain
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from .utils.gnn import create_norm, GIN, GAT, GCN
from .base import BaseMethod
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import dgl
from functools import partial
import torch.nn.functional as F


def EncoderDecoder(GNN, in_channels=1433, hidden_channels=512, enc_dec="encoding",  num_hidden=256,  num_layers=2,
                   dropout=0.2, activation='prelu', residual=False, norm=None, nhead=4, nhead_out=1, attn_drop=0.1,
                   negative_slope=0.2, concat_out=True) -> nn.Module:
    if GNN == "gat":
        mod = GAT(
            in_dim=in_channels,
            num_hidden=num_hidden,
            out_dim=hidden_channels,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif GNN == "gin":
        mod = GIN(
            in_dim=in_channels,
            num_hidden=num_hidden,
            out_dim=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif GNN == "gcn":
        mod = GCN(
            in_dim=in_channels,
            num_hidden=num_hidden, 
            out_dim=hidden_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif GNN == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_channels, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, hidden_channels)
        )
    elif GNN == "linear":
        mod = nn.Linear(in_channels, hidden_channels)
    else:
        raise NotImplementedError
    mod.dec_in_dim = in_channels
    mod.num_layers = num_layers
    mod.out_dim = hidden_channels
    mod.type = GNN
    return mod


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def setup_loss_fn(loss_fn, alpha_l):
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        raise NotImplementedError
    return criterion


def collate_fn(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1].clone().detach() for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    return batch_g, labels

class GraphMAE(BaseMethod):
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 hidden_channels: int,
                 concat_hidden: bool = False,
                 loss_function: Optional[torch.nn.Module] = None,
                 argument = None) -> None:
        super().__init__(encoder=encoder, loss_function=loss_function)
        # build encoder
        self.encoder = encoder

        # build decoder for attribute prediction
        self.decoder = decoder
        dec_in_dim = decoder.dec_in_dim
        num_layers = encoder.num_layers
        in_dim = decoder.out_dim

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.criterion = loss_function
        self._output_hidden_size = hidden_channels
        self._mask_rate = argument.model.mask_rate
        self._replace_rate = argument.model.replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._drop_edge_rate = argument.model.drop_edge_rate
        self._concat_hidden = concat_hidden
        self._decoder_type = decoder.type
        self._encoder_type = encoder.type

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = max(int(self._replace_rate * num_mask_nodes), 1)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, data):

        g = data[0]
        x = data[0].ndata['attr']
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = self.drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)
        loss = self.get_loss(x, recon, mask_nodes)
        return loss

    def get_loss(self, x, recon, mask_nodes):
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def drop_edge(self, graph, drop_rate, return_edges=False):
        if drop_rate <= 0:
            return graph

        n_node = graph.num_nodes()
        edge_mask = self.mask_edge(graph, drop_rate)
        src = graph.edges()[0]
        dst = graph.edges()[1]

        nsrc = src[edge_mask]
        ndst = dst[edge_mask]

        ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
        ng = ng.add_self_loop()

        dsrc = src[~edge_mask]
        ddst = dst[~edge_mask]

        if return_edges:
            return ng, (dsrc, ddst)
        return ng

    def mask_edge(self, graph, mask_prob):
        E = graph.num_edges()

        mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        return mask_idx

    def evaluation(self, pooler, dataloader):
        self.eval()
        x_list = []
        y_list = []
        with torch.no_grad():
            for i, (batch_g, labels) in enumerate(dataloader):
                batch_g = batch_g.to(self.device)
                feat = batch_g.ndata["attr"]
                out = self.encoder(batch_g, feat)
                out = pooler(batch_g, out)

                y_list.append(labels.numpy())
                x_list.append(out.cpu().numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        test_acc, test_std = self.evaluate_graph_embeddings_using_svm(x, y)
        print(f"#Test_acc: {test_acc:.4f}±{test_std:.4f}")
        # return x, y


    def evaluate_graph_embeddings_using_svm(self, embeddings, labels):
        result = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        for train_index, test_index in kf.split(embeddings, labels):
            x_train = embeddings[train_index]
            x_test = embeddings[test_index]
            y_train = labels[train_index].ravel()
            y_test = labels[test_index].ravel()
            params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
            svc = SVC(random_state=42)
            clf = GridSearchCV(svc, params)
            clf.fit(x_train, y_train)

            preds = clf.predict(x_test)
            # f1 = f1_score(y_test, preds, average="micro")
            test_acc = accuracy_score(y_test, preds)
            # test_acc = (torch.sum(preds == y_test).float() / y_test.shape[0]).detach().cpu().numpy()
            result.append(test_acc)
        test_acc = np.mean(result)
        test_std = np.std(result)
        return test_acc, test_std


def load_graph_classification_dataset(dataset_name, raw_dir=None, deg4feat=False):
    from dgl.data import TUDataset
    import torch.nn.functional as F
    from collections import Counter
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name, raw_dir=raw_dir)
    graph, _ = dataset[0]
    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = int(max(feature_dim, g.ndata["node_labels"].max().item()))

            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1).long()
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES

                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])

    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, feature_dim

def apply_data_augment(self, batch):
    raise NotImplementedError

def apply_emb_augment(self, h_pos):
    raise NotImplementedError


