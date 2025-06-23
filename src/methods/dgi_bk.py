import torch

from .base import BaseMethod, ContrastiveMethod
from .utils import AvgReadout
from src.augment import ShuffleNode, Echo, AugmentorDict
from src.losses import NegativeMI

from typing import Optional, Callable, Union
from src.typing import AugmentType


class DGI(BaseMethod):
    r""" Deep Graph Infomax (DGI).

    Args:
        encoder (Optional[torch.nn.Module]): the encoder to be trained.
        hidden_channels (int): output dimension of the encoder.
        readout (Union[Callable, torch.nn.Module]): the readout function to obtain the summary emb. of the entire graph.
            (default: AvgReadout())
        corruption (OptAugment): data augmentation/corruption to generate negative node pairs.
            (default: ShuffleNode())
        loss_function (Optional[torch.nn.Module]): the loss function. If None, then use the NegativeMI loss.
    """
    def __init__(self,
                 encoder: torch.nn.Module,
                 hidden_channels: int,
                 readout: Union[Callable, torch.nn.Module] = AvgReadout(),
                 corruption: AugmentType = ShuffleNode(),
                 loss_function: Optional[torch.nn.Module] = None) -> None:
        loss_function = loss_function if loss_function else NegativeMI(hidden_channels)
        super().__init__(encoder=encoder, data_augment=corruption, loss_function=loss_function)

        self.readout = readout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch2 = self.data_augment(batch).to(self._device)

        h_pos = self.encoder(batch)
        h_neg = self.encoder(batch2)

        s = self.readout(h_pos, keepdim=True)
        s = self.sigmoid(s)

        s = s.expand_as(h_pos)

        loss = self.loss_function(x=s, y=h_pos, x_ind=s, y_ind=h_neg)
        return loss
