from typing import List, Callable

from torch_geometric.transforms import BaseTransform


class TransformList(BaseTransform):
    r"""A list of transform functions."""
    def __init__(self, transform_list: List[Callable]):
        self.transform_list = transform_list

    def __call__(self, data):
        for transform in self.transform_list:
            data = transform(data)
        return data
