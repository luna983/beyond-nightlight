import torch
import warnings

from torch.utils.data import DataLoader

from .googleearthpro import GoogleEarthProInstSeg

def collate_fn(batch):
    """Helper function to collate lists of image/target pairs into a batch."""
    return tuple(zip(*batch))

def make_data_loader(cfg, modes=['train', 'val'], **kwargs):
    """Make data loaders with different datasets.

    Args:
        cfg (Config object): configurations.
        modes (list of str): list of data loader modes to be executed.
    """

    if torch.__version__ < '1.1.0':
        warnings.warn("PyTorch version below 1.1.0. pin_memory option not available.")

    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available.")
    
    data_loaders = []

    for mode in modes:
        
        # link cfg.dataset and dataloader class
        if cfg.dataset == 'googleearthpro':
            data_set = GoogleEarthProInstSeg(cfg, mode=mode)
        else:
            raise NotImplementedError
        
        # shuffle data for training set but not validation and inference
        if mode in ['train']:
            data_loader = DataLoader(
                data_set, shuffle=True,
                collate_fn=collate_fn, pin_memory=True,
                **kwargs)
        elif mode in ['val', 'infer']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_fn, pin_memory=True,
                **kwargs)
        else:
            raise NotImplementedError

        data_loaders.append(data_loader)

    return data_loaders
