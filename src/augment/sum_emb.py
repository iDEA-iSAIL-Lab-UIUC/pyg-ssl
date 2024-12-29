import torch
from torch_geometric.typing import Tensor

from .base import Augmentor
from typing import Callable, Optional


class SumEmb(Augmentor):
    """
    Get the summary for a batch of embedidngs.
    """
    def __init__(self, 
                 pooler: str="avg",
                 act: Optional[Callable]=None):
        super().__init__()
        assert pooler in ["avg", "max"], 'Pooler should be either "avg" or "max"'
        self.pooler = pooler
        self.act = act

    def __call__(self, 
                 x: Tensor, 
                 dim: int=1, 
                 keepdim: bool=False):
        if self.pooler == "avg":
            s = torch.mean(x, dim, keepdim=keepdim)
        else:
            s = torch.max(x, dim, keepdim=keepdim)
        
        if self.act:
            s = self.act(s)
        return s        
