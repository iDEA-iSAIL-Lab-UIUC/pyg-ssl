import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier as rf_cl


class RandomForestClassifier(BaseEvaluator):
    def __init__(self,
                 search: bool = False,
                 n_estimators: int = 1,
                 criterion: str = 'gini',
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[int, float, str, None] = 'auto',
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 obb_score: bool = False,
                 n_jobs: int = 1,
                 random_state: Union[int, None] = None,
                 verbose: int = 0,
                 warm_start: bool = False,
                 class_weight: dict = None,
                 n_run: int = 50,
                 device: Union[str, int] = "cuda") -> None:

        self.search = search
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.verbose = verbose
        self.obb_score = obb_score
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_run = n_run
        self.device = device

    def __call__(self, embs, dataset):
        r"""
        TODO: maybe we need to return something.
        """
        val_accs, test_accs = [], []
        embs, labels = embs.detach().cpu().numpy(), dataset.y.detach().cpu().numpy()

        for n in range(self.n_run):
            n_splits, train_mask, val_mask, test_mask = process_split(dataset=dataset)
            test_mask = dataset.test_mask
            train_mask, val_mask, test_mask = train_mask.detach().cpu().numpy(), val_mask.detach().cpu().numpy(), test_mask.detach().cpu().numpy()
            for i in range(n_splits):
                train_msk = train_mask[i]
                val_msk = val_mask[i]
                # some datasets (e.g., wikics) with multiple splits only has one test mask
                test_msk = test_mask if len(test_mask.shape) == 1 else test_mask[i]

                val_acc, test_acc = self.single_run(
                    embs=embs, labels=labels, train_mask=train_msk, val_mask=val_msk, test_mask=test_msk)

                val_accs.append(val_acc * 100)
                test_accs.append(test_acc * 100)

        val_accs = np.stack(val_accs)
        test_accs = np.stack(test_accs)

        val_acc, val_std = val_accs.mean(), val_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('Evaluate node classification results')
        print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(val_acc, val_std, test_acc, test_std))

    def single_run(self, embs, labels, train_mask, val_mask, test_mask) -> (np.ndarray, np.ndarray):
        val_accs, test_accs = [], []
        # emb_dim, num_class = embs.shape[1], labels.unique().shape[0]

        # embs, labels = embs.detach().cpu().numpy(), labels.detach().cpu().numpy()
        if self.search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(rf_cl(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                                                max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap, oob_score=self.obb_score, 
                                                n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start, class_weight=self.class_weight), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = rf_cl(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                                                max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap, oob_score=self.obb_score, 
                                                n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start, class_weight=self.class_weight)
        classifier.fit(embs[train_mask], labels[train_mask])

        val_accs.append(accuracy_score(labels[val_mask], classifier.predict(embs[val_mask])))
        test_accs.append(accuracy_score(labels[test_mask], classifier.predict(embs[test_mask])))
    
        return np.mean(val_accs), np.mean(test_accs)
    
def process_split(dataset):
    r"""If the dataset only has one split, then add an additional dimension to the train_mask and val_mask.
        If the dataset has multiple splits, then return the number of splits, original train_mask and val_mask.
    """
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    if len(dataset.train_mask.shape) > 1:
        n_splits = dataset.train_mask.shape[0]
    else:
        n_splits = 1
        train_mask = torch.unsqueeze(dataset.train_mask, 0)
        val_mask = torch.unsqueeze(dataset.val_mask, 0)
        test_mask = torch.unsqueeze(dataset.test_mask, 0)
    return n_splits, train_mask, val_mask, test_mask

