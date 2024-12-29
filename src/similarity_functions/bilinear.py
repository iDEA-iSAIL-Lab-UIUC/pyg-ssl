import torch
from torch import nn


class Bilinear(nn.Module):
    def __init__(self, in_channels: int=512):
        super().__init__()
        self.bilinear = nn.Bilinear(in_channels, in_channels, 1)

    def forward(self, x, y):
        """
        Args:
            x: [N, *, dim]
            y: [N, *, dim]
            * means all but the last dimension should be the same.
        
        Returns:
            similarity scores of (x, y): [N, *, 1]
        """
        scores = self.bilinear(x, y)
        return scores
