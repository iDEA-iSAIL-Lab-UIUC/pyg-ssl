import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from typing import List, Union, Optional, Any
from torch_geometric.typing import Tensor

# based on torch_geometric.transforms.normalize_features.py
# from torch_geometric.transforms.normalize_features import NormalizeFeatures


@functional_transform('normalize_features_ssl')
class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
        ord (Optional[str, int, float]): The order of normalization.
            (default: "sum1")
            sum1: x = x - min(x), x / sum(x)
            other supported normalization:
            https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
    """

    ord = "sum1"

    def __init__(self, attrs: List[str] = ["x"], ord: Optional[Any] = None):
        self.attrs = attrs
        if ord is not None:
            self.ord = ord

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if self.ord == "sum1":
                    store[key] = sum1(value)
                else:
                    denominator = torch.linalg.norm(value, ord=self.ord, dim=-1, keepdim=True)
                    value = value.div_(denominator)
                    value[torch.isinf(value)] = 0.
                    store[key] = value
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def sum1(value: Tensor):
    value = value - value.min()
    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return value
