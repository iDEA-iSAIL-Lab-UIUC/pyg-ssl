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

class SimpleTrainer(BaseTrainer):
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
        self.device = device
        self.early_stopper = EarlyStopper(patience=self.patience)

    def train(self):
        self.method = self.method.to(self.device)
        new_loader = self.method.apply_data_augment_offline(self.data_loader)
        if new_loader != None:
            self.data_loader = new_loader
        for epoch in range(self.n_epochs):
            start_time = time.time()

            for data in self.data_loader:
                self.method.train()
                self.optimizer.zero_grad()

                data = self.push_batch_to_device(data)
                loss = self.method(data)

                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            info = "Epoch {}: loss: {:.4f}, time: {:.4f}s".format(epoch, loss.detach().cpu().numpy(), end_time-start_time)
            print(info)

            self.early_stopper.update(loss)  # update the status
            if self.early_stopper.save:
                self.save()
            if self.early_stopper.stop:
                return

    # push data to device
    def push_batch_to_device(self, batch):
        if type(batch) is tuple:
            f = lambda x: tuple(x_.to(self.device) for x_ in batch)
            return f(batch)
        else:
            return batch.to(self.device)

    def check_dataloader(self, dataloader):
        assert hasattr(dataloader, 'x'), 'The dataset does not have attributes x.'

