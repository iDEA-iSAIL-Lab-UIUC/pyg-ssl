import torch

from typing import Optional
from torch_geometric.typing import Tensor

from src.similarity_functions import Bilinear


class NegativeMI(torch.nn.Module):
    r"""Negative Mutual Information.
    Estimate the negative mutual information loss for the specified neural similarity function (sim_function) and the
    inputs x, y, x_ind, y_ind. (x, y) is sampled from the joint distribution (x, y)~p(x, y). x_ind and y_ind are
    independently sampled from their marginal distributions x_ind~p(x), y_ind~p(y).
    Reference: Deep Graph Infomax.

    Args:
        in_channels (Optional[int]): input channels to the neural mutual information estimator.

        sim_func (Optional[torch.nn.Module]): the neural similarity measuring function.
        The default is the bilinear similarity function used by DGI.
    """
    def __init__(self,
                 in_channels: Optional[int] = None,
                 sim_func: Optional[torch.nn.Module] = None):
        super().__init__()
        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self.sim = sim_func
        if self.sim is None:
            assert in_channels is not None, \
                "If use the default bilinear discriminator, then hidden_channels must be set."
            self.sim = Bilinear(in_channels=in_channels)

    def forward(self, x: Tensor, y: Tensor, x_ind: Tensor, y_ind: Tensor):
        r"""
        Args:
            (x, y): sampled from the joint distribution p(x, y). [batch_size, hidden_channels]
            x_ind: sampled from p(x).  [batch_size, hidden_channels]
            y_ind: sampled from p(y).  [batch_size, hidden_channels]

        Returns:
            The loss of the mutual information estimation.
        """
        logits_pos = self.sim(x=x, y=y)
        logits_neg = self.sim(x=x_ind, y=y_ind)
        logits = torch.cat([logits_pos, logits_neg], -1)

        label_pos = torch.ones(logits_pos.shape[0])
        label_neg = torch.zeros(logits_neg.shape[0])
        labels = torch.stack((label_pos, label_neg), -1).to(x.device)
        return self.loss_func(logits, labels)
