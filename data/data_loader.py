import copy
import pickle
import random
from functools import partial
from typing import List
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from utilities.configs import DatasetTypes, DatasetUsages, ModelTypes, DataDict
from data.semantic_kitti.dataset import KittiDataset


def reset_seed_worker_init_fn(worker_id):
    """Reset seed for data loader worker."""
    # Use either one:
    # -----------------------------------------
    # seed = torch.initial_seed() % (2 ** 32)
    # np.random.seed(seed)
    # random.seed(seed)
    # -----------------------------------------
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)
    # -----------------------------------------

def collate_fn_stack_mode(batch):
    data = {}
    for key in batch[0].keys():
        for idx, input_dict in enumerate(batch):
            if key not in data.keys():
                data[key] = []
            if isinstance(input_dict[key], torch.Tensor) or key == DataDict.name:
                data[key].append(input_dict[key])
            else:
                data[key].append(torch.from_numpy(input_dict[key]))
    for key in data.keys():
        if key == DataDict.name:
            pass
        else:
            data[key] = torch.stack(data[key])
    return data


def get_data_loader(cfg, usages=DatasetUsages.train, model_type=ModelTypes.cvae):
    dataset = KittiDataset(config_path=cfg.config_dir, scale=cfg.scale, root=cfg.data_root,
                           dataset_usage=usages, model_type=model_type)
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        sampler=sampler,
        collate_fn=partial(collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader


def train_data_loader(cfg, model_type=ModelTypes.cvae):
    """
    This function is to create a training dataloader with pytorch interface
    Args:
        model_type:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfgs, usages=DatasetUsages.train, model_type=model_type)


def evaluation_data_loader(cfg, model_type=ModelTypes.cvae, distribute_evaluation=True):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        model_type:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    if not distribute_evaluation:
        cfgs.distributed = False
    return get_data_loader(cfg=cfgs, usages=DatasetUsages.val, model_type=model_type)


def test_data_loader(cfg, model_type=ModelTypes.cvae):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfgs, usages=DatasetUsages.test, model_type=model_type)
