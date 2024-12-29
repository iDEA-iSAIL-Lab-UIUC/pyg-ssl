import torch
from src.typing import Tensor


class BilinearSim(torch.nn.Module):
    r"""Bilinear similarity between two tensors."""
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.sim = torch.nn.Bilinear(in_channels, in_channels, 1)

    def forward(self, x: Tensor, y: Tensor):
        r"""
        Args:
            x: [batch_size, dim]
            y: [batch_size, dim]

        Returns:
            similarity scores of (x, y): [batch_size, 1]
        """
        logits = self.sim(x, y)
        return logits
