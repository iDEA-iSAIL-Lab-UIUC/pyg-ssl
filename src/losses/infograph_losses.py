'''
    Par of code are adapted from cortex_DIM
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LocalGlobalLoss(torch.nn.Module):
    r"""Negative Mutual Information.
    Estimate the negative mutual information loss for the specified neural similarity function (sim_function) and the
    inputs x, y, x_ind, y_ind. (x, y) is sampled from the joint distribution (x, y)~p(x, y). x_ind and y_ind are
    independently sampled from their marginal distributions x_ind~p(x), y_ind~p(y).
    Reference: Deep Graph Infomax.

    Args:
        in_channels (Optional[int]): input channels to the neural mutual information estimator.

        sim_function (Optional[torch.nn.Module]): the neural similarity measuring function.
        The default is the bilinear similarity function used by DGI.
    """
    def __init__(self, ):
        super().__init__()

    def forward(self, l_enc, g_enc, batch, measure):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
        num_graphs = g_enc.shape[0]
        num_nodes = l_enc.shape[0]

        pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
        neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
        for nodeidx, graphidx in enumerate(batch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        res = torch.mm(l_enc, g_enc.t())

        E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
        E_pos = E_pos / num_nodes
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))

        return E_neg - E_pos


class AdjLoss(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, l_enc, edge_index):
        num_nodes = l_enc.shape[0]

        adj = torch.zeros((num_nodes, num_nodes)).cuda()
        mask = torch.eye(num_nodes).cuda()
        for node1, node2 in zip(edge_index[0], edge_index[1]):
            adj[node1.item()][node2.item()] = 1.
            adj[node2.item()][node1.item()] = 1.

        res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
        res = (1-mask) * res
        # print(res.shape, adj.shape)
        # input()

        return self.loss(res, adj)


def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))

def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq
