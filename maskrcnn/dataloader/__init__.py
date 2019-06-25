from torch.utils.data import DataLoader

from .googleearthpro import GoogleEarthProInstSeg

def make_data_loader(cfg, modes=['train', 'val'], **kwargs):
    """Make data loaders with different datasets.

    Args:
        cfg (Config object): configurations.
        modes (list of str): list of data loader modes to be executed.
    """

    data_loaders = []

    for mode in modes:
        
        # link cfg.dataset and dataloader class
        if cfg.dataset == 'googleearthpro':
            data_set = GoogleEarthProInstSeg(cfg, mode=mode)
        else:
            raise NotImplementedError
        
        # shuffle data for training set but not validation and inference
        if mode in ['train']:
            data_loader = DataLoader(data_set, shuffle=True, **kwargs)
        elif mode in ['val', 'infer']:
            data_loader = DataLoader(data_set, shuffle=False, **kwargs)
        else:
            raise NotImplementedError

        data_loaders.append(data_loader)

    return data_loaders