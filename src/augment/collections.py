"""TODO: looks like this file is useless."""

from .base import *
from .shuffle_node import ShuffleNode
from .random_drop_edge import RandomDropEdge
from .negative import ComputePPR, ComputeHeat
from .shuffle_node import ShuffleNode

__all__ = [
    "augment_dgi",
    "augment_mvgrl_ppr",
    "augment_mvgrl_heat",
    "augment_bgrl_1",
    "augment_bgrl_2",
]

# augment_dgi = DataShuffle(is_x=True)
augment_dgi = ShuffleNode()
augment_mvgrl_ppr = ComputePPR()
augment_mvgrl_heat = ComputeHeat()
augment_bgrl_1 = RandomDropEdge()
augment_bgrl_2 = RandomDropEdge()
