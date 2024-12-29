from torch_geometric.typing import *
from src.augment import Augmentor, AugmentorList, AugmentorDict

AugmentType = Union[Augmentor, AugmentorList, AugmentorDict]
OptAugment = Optional[AugmentType]
