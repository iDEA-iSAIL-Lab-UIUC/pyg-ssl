import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC


class SVCRegression(BaseEvaluator):
    def __init__(self,
                 C: float = 1.0,
                 search: bool = True,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str='auto',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 probability: bool = False,
                 tol: float = 0.001,
                 cache_size: int = 200,
                 class_weight: dict = None,
                 verbose: bool = False,
                 max_iter: int = -1,
                 decision_function_shape: str = 'ovr',
                 random_state: int = None,
                 n_run: int = 50,
                 device: Union[str, int] = "cuda") -> None:
        self.C = C
        self.search = search
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shirinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
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
        if self.search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, shrinking=self.shirinking, probability=self.probability,
                        tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, 
                        decision_function_shape=self.decision_function_shape,random_state=self.random_state), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, shrinking=self.shirinking, probability=self.probability,
                        tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, 
                        decision_function_shape=self.decision_function_shape,random_state=self.random_state)
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

