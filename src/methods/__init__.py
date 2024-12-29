from .base import BaseMethod

from .dgi import DGI, DGIEncoder
from .graphcl import GraphCL, GraphCLEncoder
from .infograph import InfoGraph
from .bgrl import BGRL, BGRLEncoder
from .afgrl import AFGRL, AFGRLEncoder
from .merit import GCN, Merit
from .sugrl import SUGRL, SugrlGCN, SugrlMLP
from .mvgrl import MVGRL, MVGRLEncoder

from .hdmi import HDMI, HDMIEncoder
from .heco import HeCo, Sc_encoder, Mp_encoder, HeCoDBLPTransform

__all__ = [
    "BaseMethod",
    "DGI",
    "DGIEncoder",
    "GraphCL",
    "GraphCLEncoder",
]
