import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor
from sklearn.metrics import pairwise

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC


class SimSearch(BaseEvaluator):
    def __init__(self,
                 sim_list: list = [5, 10, 20, 50, 100],
                 n_run: int = 50,
                 device: Union[str, int] = "cuda") -> None:
        self.sim_list = sim_list
        self.n_run = n_run
        self.device = device

    def __call__(self, embs, dataset):
        r"""
        TODO: maybe we need to return something.
        """
        embs, labels = embs.detach().cpu().numpy(), dataset.y.detach().cpu().numpy()
        st = self.single_run(embs=embs, labels=labels)
        st = ','.join(st)

        print('Evaluate node classification results')
        print("\t[Similarity]" + " : [{}]".format(st))
       
    def single_run(self, embs, labels) -> (np.ndarray):
        numRows = embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(embs) - np.eye(numRows)
        st = []
      
        for N in self.sim_list:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(labels, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(labels, N).reshape(numRows,N)
            st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))
    
        return st