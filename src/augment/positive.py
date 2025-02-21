"""TODO: remove this file."""

from .base import Augmentor


class Echo(Augmentor):
    """ TODO: maybe this class is unnecessary. """
    def __init__(self):
        super().__init__()

    def apply(self, data):
        return data
