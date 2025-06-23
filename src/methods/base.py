import os
import numpy
import torch

from typing import Union, Callable, Optional, Any
from src.typing import OptAugment, Tensor

ENCODER_NAME = "encoder.ckpt"
MODEL_NAME = "model.ckpt"


class BaseMethod(torch.nn.Module):
    r"""Base class for self-supervised learning methods.

    Args:
        encoder (torch.nn.Module): the encoder to be trained.
        loss_function (Callable): loss function.
        data_augment (OptAugment): data augment to be used.
        emb_augment (OptAugment): embedding augment to be used.
    """

    def __init__(self,
                 encoder: torch.nn.Module,
                 device: str="cuda",
                 save_root: str="./",
                 *args, **kwargs):
        super().__init__()

        self.encoder = encoder
        self._device = device  # device: cuda or cpu
        self.save_root = save_root  # record the latest path used by save() or load()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Perform the forward pass."""
        raise NotImplementedError

    def apply_data_augment(self, *args, **kwargs) -> Any:
        r"""Apply online data augmentation. 
            These augmentations are used online within methods."""
        raise NotImplementedError

    def apply_data_augment_offline(self, *args, **kwargs):
        r"""Apply offline data augmentation. 
            These augmentations are used offline in the trainer."""
        return None

    def apply_emb_augment(self, *args, **kwargs) -> Any:
        r"""Apply online embedding augmentation."""
        raise NotImplementedError

    def get_loss(self, *args, **kwargs) -> Tensor:
        r"""Get loss."""
        raise NotImplementedError

    def save(self, path: Optional[str] = None) -> None:
        r"""Save the parameters of the entire model to the specified path."""
        if path is None:
            path = self.__class__.__name__ + ".ckpt"
        path = os.path.join(self.save_root, path)
        torch.save(self.state_dict(), path)

    def load(self, path: Optional[str] = None) -> None:
        r"""Load the parameters from the specified path."""
        if path is None:
            raise FileNotFoundError
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.to(self.device)
