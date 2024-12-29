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
        # if config:
        #     self.optimizer = torch.optim.Adam(self.method.parameters(), lr, weight_decay=config.optim.weight_decay)

        self.optimizer = torch.optim.AdamW(self.method.parameters(), lr, weight_decay=weight_decay)
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.patience = patience
        self.device = device
        # scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
        #             else ( 1 + np.cos((epoch-1000) * np.pi / (n_epochs - 1000))) * 0.5
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = scheduler)
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
                # self.scheduler.step()

            end_time = time.time()
            info = "Epoch {}: loss: {:.4f}, time: {:.4f}s".format(epoch, loss.detach().cpu().numpy(), end_time-start_time)
            print(info)

            # # ------------------ Evaluator -------------------
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # data_pyg = self.data_loader.dataset.data.to(device)
            # y, embs = self.method.get_embs(self.data_loader)
            # data_pyg.x = embs
            # from src.evaluation import LogisticRegression
            # lg = LogisticRegression(lr=0.001, weight_decay=0, max_iter=100, n_run=1, device=device)
            # lg(embs=embs, dataset=data_pyg)


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
        # assert hasattr(dataloader, 'train_mask'), 'T'
        # return 0
