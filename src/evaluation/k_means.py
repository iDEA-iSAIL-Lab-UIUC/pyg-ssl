import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor
from sklearn.metrics import pairwise

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


class K_Means(BaseEvaluator):
    def __init__(self,
                 k: int = 8,
                 average_method: str = "arithmetic",
                 init = "k-means++",
                 n_init = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state = None,
                 copy_x: bool = True,
                 algorithm: str = "auto",
                 n_run: int = 50,
                 device: Union[str, int] = "cuda") -> None:
        self.k = k
        self.average_method = average_method
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.n_run = n_run
        self.device = device

    def __call__(self, embs, dataset):
        r"""
        TODO: maybe we need to return something.
        """
        embs, labels = embs.detach().cpu().numpy(), dataset.y.detach().cpu().numpy()
        NMI_list = []

        for n in range(self.n_run):
           
            s = self.single_run(
                embs=embs, labels=labels)
            NMI_list.append(s)

        mean = np.mean(NMI_list)
        std = np.std(NMI_list)

        print('Evaluate node classification results')
        print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))

    def single_run(self, embs, labels) -> (np.ndarray):

        estimator = KMeans(n_clusters=self.k, init=self.init, n_init=self.n_init, max_iter=
                           self.max_iter, tol=self.tol, verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x, algorithm=self.algorithm)
        estimator.fit(embs)
        y_pred = estimator.predict(embs)
        s = normalized_mutual_info_score(labels, y_pred, average_method=self.average_method)
        
        return s
