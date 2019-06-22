from torch.utils.data import DataLoader
from googleearthpro import GoogleEarthProInstSeg


def make_data_loader(cfg, **kwargs):

    if cfg.dataset == 'googleearthpro':
        train_set = GoogleEarthProInstSeg(cfg, mode='train')
        val_set = GoogleEarthProInstSeg(cfg, mode='val')
        train_loader = DataLoader(train_set, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, shuffle=False, **kwargs)
        return train_loader, val_loader

    else:
        raise NotImplementedError

