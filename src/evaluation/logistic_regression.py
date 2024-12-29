import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor

from tqdm import tqdm


class LogisticRegression(BaseEvaluator):
    def __init__(self,
                 lr: float = 0.01,
                 weight_decay: float = 0.,
                 max_iter: int = 100,
                 n_run: int = 50,
                 device: Union[str, int] = "cuda") -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.n_run = n_run
        self.device = device

    def __call__(self, embs, dataset):
        r"""
        TODO: maybe we need to return something.
        """
        val_accs, test_accs = [], []

        for n in tqdm(range(self.n_run)):
            n_splits, train_mask, val_mask, test_mask = process_split(dataset=dataset)
            test_mask = dataset.test_mask
            for i in range(n_splits):
                train_msk = train_mask[i]
                val_msk = val_mask[i]
                # some datasets (e.g., wikics) with multiple splits only has one test mask
                test_msk = test_mask if type(test_mask) == torch.Tensor and len(test_mask.shape) == 1 else test_mask[i]
                val_acc, test_acc = self.single_run(
                    embs=embs, labels=dataset.y, train_mask=train_msk, val_mask=val_msk, test_mask=test_msk)

                val_accs.append(val_acc * 100)
                test_accs.append(test_acc * 100)

        val_accs = np.stack(val_accs)
        test_accs = np.stack(test_accs)

        val_acc, val_std = val_accs.mean(), val_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('Evaluate node classification results')
        print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(val_acc, val_std, test_acc, test_std))

    def single_run(self, embs, labels, train_mask, val_mask, test_mask):
        emb_dim, num_class = embs.shape[1], labels.unique().shape[0]

        embs, labels = embs.to(self.device), labels.to(self.device)
        classifier = LogisticRegressionClassifier(emb_dim, num_class).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for iter in range(self.max_iter):
            classifier.train()
            logits, loss = classifier(embs[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_logits, _ = classifier(embs[val_mask], labels[val_mask])
        test_logits, _ = classifier(embs[test_mask], labels[test_mask])
        val_preds = torch.argmax(val_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        val_acc = (torch.sum(val_preds == labels[val_mask]).float() /
                   labels[val_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == labels[test_mask]).float() /
                    labels[test_mask].shape[0]).detach().cpu().numpy()
        return val_acc, test_acc


class LogisticRegressionClassifier(nn.Module):
    r"""This is the logistic regression classifier used by DGI.

    Args:
        num_dim (int): the size of embedding dimension.
        num_class (int): the number of classes.
    """
    def __init__(self, num_dim: int, num_class: int):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, y: Tensor):
        r"""
        Args:
            x (Tensor): embeddings
            y (Tensor): labels
        """
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss


def process_split(dataset):
    r"""If the dataset only has one split, then add an additional dimension to the train_mask and val_mask.
        If the dataset has multiple splits, then return the number of splits, original train_mask and val_mask.
    """
    if not hasattr(dataset, 'train_mask'):
        from sklearn.model_selection import StratifiedKFold
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
        dataset.train_mask = []
        dataset.val_mask = []
        dataset.test_mask = []
        for train_index, test_index in kf.split(dataset.x.cpu(), dataset.y.cpu()):
            dataset.train_mask.append(torch.LongTensor(train_index))
            dataset.test_mask.append(torch.LongTensor(test_index))
            dataset.val_mask.append(torch.LongTensor(test_index))
        n_splits = len(dataset.train_mask)
        return n_splits, dataset.train_mask, dataset.val_mask, dataset.test_mask
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    if type(train_mask) == torch.Tensor and len(train_mask.shape) > 1:
        n_splits = dataset.train_mask.shape[0]
    elif type(dataset.train_mask) == list and len(dataset.train_mask) > 1:
        n_splits = len(dataset.train_mask)
    else:
        n_splits = 1
        train_mask = torch.unsqueeze(dataset.train_mask, 0)
        val_mask = torch.unsqueeze(dataset.val_mask, 0)
        test_mask = torch.unsqueeze(dataset.test_mask, 0)
    return n_splits, train_mask, val_mask, test_mask
