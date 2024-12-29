from torch import Tensor
import torch
import torch.nn.functional as F


class SIGLoss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (x * y).sum(1)
        loss = torch.sigmoid(-loss)
        loss = loss.mean()
        return loss

