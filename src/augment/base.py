from collections import UserDict
from typing import List, Union


class Augmentor:
    r"""Base class for augmentor."""
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AugmentorList:
    r"""A wrapper for a list of augmentors. Sequentially apply each augmentor to inputs."""
    def __init__(self, augmentors: Union[Augmentor, List[Augmentor]]):
        if type(augmentors) == list:
            self.augmentors = augmentors
        else:
            self.augmentors = [augmentors]

    def __call__(self, inputs):
        for augmentor in self.augmentors:
            inputs = augmentor(inputs)
        return inputs

    # def append(self, item):
    #     return self.augmentors.append(item)

class AugmentorDict(UserDict):
    r"""Base class for augmentation. A dictionary of augmentors."""
    def __setitem__(self, key, value):
        try:
            assert isinstance(value, Augmentor) or isinstance(value, AugmentorList)
        except AssertionError:
            raise "AssertionError. The value should be an instance of Augmentor or Augmentors."
        super().__setitem__(key, value)
