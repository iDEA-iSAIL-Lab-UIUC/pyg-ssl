import torch
import torch.nn.functional as F


def dot(x, y):
    """
    Args:
        x: [batch, n1, dim]
        y: [batch, n2, dim]

    Returns:
        [batch, n1, n2]
    """
    assert len(x.shape) == 3 and len(x.shape) == 3, \
        "The shape of x and y should be "    
    return torch.bmm(x, torch.transpose(y, 1, 2))


def cosine(x, y):
    """
    Args:
        x: [batch, n1, dim]
        y: [batch, n2, dim]

    Returns:
        [batch, n1, n2]
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return dot(x, y)


def distance(x, y, p=2):
    """
    Args:
        x: [batch, n1, dim]
        y: [batch, n2, dim]
        p: the order

    Returns:
        [batch, n1, n2]
    """
    x = x.contiguous()
    y = y.contiguous()
    return -torch.cdist(x, y, p=p)
