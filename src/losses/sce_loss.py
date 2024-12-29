from torch import Tensor
import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()
        return loss
