from .base import BaseEvaluator
from typing import Union
from src.typing import Tensor

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


class TSNEVisulization(BaseEvaluator):
    def __init__(self,
                 n_colors: int = 8,
                 n_components: int =2,
                 perplexity: float = 30.0,
                 early_exaggeration: float = 12.0,
                 learning_rate: Union[float, str] = "auto",
                 n_iter: int = 1000,
                 n_iter_without_progress: int = 300,
                 min_grad_norm: float = 1e-7,
                 metric: str = 'euclidean',
                 metric_params: dict = None,
                 init: Union[str, np.ndarray] = "pca",
                 verbose: int = 0,
                 random_state: int = None,
                 method: str = 'barnes_hut',
                 angle: float = 0.5,
                 n_jobs: int = None,
                 device: Union[str, int] = "cuda") -> None:
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.n_colors = n_colors
        self.device = device

    def __call__(self, embs, dataset):
        r"""
        TODO: maybe we need to return something.
        """
        print(self.method)
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, early_exaggeration=self.early_exaggeration, 
                    learning_rate=self.learning_rate, n_iter=self.n_iter, n_iter_without_progress=self.n_iter_without_progress, min_grad_norm=self.min_grad_norm, metric=self.metric, 
                    init=self.init, verbose=self.verbose, random_state=self.random_state, method=self.method, n_jobs=self.n_jobs)
       
        color = sns.hls_palette(self.n_colors)

        if isinstance(embs, Tensor):
            embs = embs.detach().cpu().numpy()
            labels = np.reshape(dataset.y.detach().cpu().numpy(), -1)
        else:
            labels = np.reshape(dataset.y, -1)
  

            
        tsne_results = tsne.fit_transform(embs)

        x = tsne_results[:, 0]
        y = tsne_results[:, 1]

        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_size_inches(10, 10)
    
        for i in range(int(np.max(labels))+1):
            idx = labels == i
            x0 = x[idx]
            y0 = y[idx]
            if self.n_components > 2:
                z0 = tsne_results[:, 2][idx]
                ax.scatter3D(x0, y0, z0, color=color[i], marker=".", linewidths=1)
            else:
                plt.scatter(x0, y0, color=color[i], marker=".", linewidths=1)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig('test.png')

    
    

        