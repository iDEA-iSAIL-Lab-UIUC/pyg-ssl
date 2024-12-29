from .base import Augmentor


class Echo(Augmentor):
    def __call__(self, inputs):
        return inputs
