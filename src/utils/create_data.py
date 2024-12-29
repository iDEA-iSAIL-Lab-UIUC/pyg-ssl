import torch
import numpy as np
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS

def create_masks(data, data_name=None):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if not hasattr(data, "train_mask"):
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
            else:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    else:  # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data
import os
import os.path as osp
def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def decide_config(root, name):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param name: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    name = name.lower()
    if name == 'cora' or name == 'citeseer' or name == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Planetoid, "src": "pyg"}
    elif name == "computers":
        name = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}        
    elif name == "photo":
        name = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}
    elif name == "cs" :
        name = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    elif name == "physics":
        name = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    elif name == "wikics":
        name = "WikiCS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": name, "class": WikiCS, "src": "pyg"}        
    else:
        raise Exception(
            f"Unknown dataset name {name}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics' 'wikics'")
    return params