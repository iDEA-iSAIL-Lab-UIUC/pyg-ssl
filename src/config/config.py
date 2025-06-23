# reference: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/graphgym/config.html

import functools
import inspect
import logging
import os
import shutil
import warnings
from collections.abc import Iterable
from dataclasses import asdict
# from typing import Any
from yacs.config import CfgNode as CN
import torch_geometric.graphgym.register as register
from torch_geometric.data.makedirs import makedirs


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #

    # Use cuda or not
    cfg.use_cuda = False

    cfg.gpu_idx = 0

    # Output directory
    cfg.out_dir = 'results'

    # Config name (in out_dir)
    cfg.cfg_dest = 'config.yaml'

    # Names of registered custom metric funcs to be used (use defaults if none)
    cfg.custom_metrics = []

    # Numpy random seed
    cfg.seed = 0

    # Pytorch random seed
    cfg.torch_seed = 0

    # Print rounding
    cfg.round = 4

    # If visualize embedding.
    cfg.view_emb = False

    # ----------------------------------------------------------------------- #
    # Globally shared variables:
    # These variables will be set dynamically based on the input dataset
    # Do not directly set them here or in .yaml files
    # ----------------------------------------------------------------------- #

    cfg.share = CN(new_allowed=True)

    # Size of input dimension
    cfg.share.dim_in = 1

    # Size of out dimension, i.e., number of labels to be predicted
    cfg.share.dim_out = 1

    # Number of dataset splits: train/val/test
    cfg.share.num_splits = 1

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN(new_allowed=True)

    # Name of the dataset
    cfg.dataset.name = 'Cora'

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = './datasets'

    # use pyg_data to load dataset
    cfg.dataset.root = 'pyg_data'

    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN(new_allowed=True)

    # Model type to use
    cfg.model.type = 'gnn'

    # Input channels
    cfg.model.in_channels = 1433

    # Hidden channels
    cfg.model.hidden_channels = 512

    # Number of layers in the Model
    cfg.model.num_layers = 1

    # Augmentation type
    cfg.aug_type = 'mask'

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN(new_allowed=True)

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.001

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 1000

    # Number of runs
    cfg.optim.run = 5

    # Set user customized cfgs
    for func in register.config_dict.values():
        func(cfg)
    return cfg


def load_yaml(filename):
    try:  # Define global config object
        cfg = CN(new_allowed=True)
    except ImportError:
        cfg = None
        warnings.warn("Could not define global config object. Please install "
                      "'yacs' via 'pip install yacs' in order to use GraphGym")
    cfg = set_cfg(cfg)
    cfg.merge_from_file(filename)
    return cfg
