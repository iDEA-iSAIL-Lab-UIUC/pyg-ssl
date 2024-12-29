from .base import Augmentor, AugmentorList, AugmentorDict
from .collections import *
from .echo import Echo
from .shuffle_node import ShuffleNode
from .augment_subgraph import AugmentSubgraph
from .random_drop_node import RandomDropNode
from .random_mask import RandomMask
from .random_drop_edge import RandomDropEdge


def concate_aug_type(augment_neg, aug):
    if augment_neg is None:
        if aug == 'edge':
            augment_neg = AugmentorList([RandomDropEdge()])
        elif aug == 'mask':
            augment_neg = AugmentorList([RandomMask()])
        elif aug == 'node':
            augment_neg = AugmentorList([RandomDropNode()])
        elif aug == 'subgraph':
            augment_neg = AugmentorList([AugmentSubgraph()])
        else:
            assert 'unrecognized augmentation method'
    else:
        if aug == 'edge':
            augment_neg.append(RandomDropEdge())
        elif aug == 'mask':
            augment_neg.append(RandomMask())
        elif aug == 'node':
            augment_neg.append(RandomDropNode())
        elif aug == 'subgraph':
            augment_neg.append(AugmentSubgraph())
        else:
            assert 'unrecognized augmentation method'
    return augment_neg
