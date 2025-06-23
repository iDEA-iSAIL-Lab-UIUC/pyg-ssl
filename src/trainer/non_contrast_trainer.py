import time
import torch
from torch_geometric.loader import DataLoader
from src.methods import BaseMethod
from .base import BaseTrainer
from .utils import EarlyStopper
from typing import Union
import numpy as np
from src.methods.utils import EMA, update_moving_average
from src.evaluation import LogisticRegression
from tqdm import tqdm

class NonContrastTrainer(BaseTrainer):
    r"""
    TODO: 1. Add descriptions.
          2. Do we need to support more arguments?
    """
    def __init__(self,
                 method: BaseMethod,
                 data_loader: DataLoader,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 n_epochs: int = 10000,
                 patience: int = 50,
                 device: Union[str, int] = "cuda:0",
                 use_ema: bool = False,
                 moving_average_decay: float = 0.9,
                 save_root: str = "./ckpt",
                 dataset=None):
        super().__init__(method=method,
                         data_loader=data_loader,
                         save_root=save_root,
                         device=device)
        self.optimizer = torch.optim.AdamW(self.method.parameters(), lr, weight_decay=weight_decay)
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.patience = patience
        self.device = devic
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_updater = EMA(moving_average_decay, n_epochs)
        self.early_stopper = EarlyStopper(patience=self.patience)

    def train(self):
        self.method = self.method.to(self.device)
        new_loader = self.method.apply_data_augment_offline(self.data_loader)
        if new_loader != None:
            self.data_loader = new_loader
        for epoch in tqdm(range(self.n_epochs)):
            start_time = time.time()

            for data in self.data_loader:
                self.method.train()
                self.optimizer.zero_grad()

                loss = self.method(data)

                loss.backward()
                self.optimizer.step()
                if self.use_ema:
                    update_moving_average(self.ema_updater, self.method.teacher_encoder, self.method.encoder)

            end_time = time.time()
            info = "Epoch {}: loss: {:.4f}, time: {:.4f}s".format(epoch, loss.detach().cpu().numpy(), end_time-start_time)
            
            if (epoch+1)%20==0:
                print(info)
                self.method.eval()
                data_pyg = self.dataset.data.to(self.method.device)
                embs = self.method.get_embs(data_pyg).detach()
                lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=100, n_run=5, device=self.device)
                lg(embs=embs, dataset=data_pyg)
            self.early_stopper.update(loss)  # update the status
            if self.early_stopper.save:
                self.save()
            if self.early_stopper.stop:
                return
