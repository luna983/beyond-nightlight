from torch.utils.data import DataLoader

from .inst_seg import InstSeg


def collate_fn(batch, mode):
    """Helper function to collate lists of image/target pairs into a batch.

    Args:
        batch (tuple): image target pairs.
        mode (str): in ['train', 'val', 'infer']

    Returns:
        tuple[N]: containing N (image, target) pairs, dropping empty images
            in training mode
    """
    if mode in ['train']:
        return tuple([(image, target) for image, target in zip(*batch)
                      if target is not None])
    elif mode in ['val', 'infer']:
        return tuple(zip(*batch))
    else:
        raise NotImplementedError


def make_data_loader(cfg, modes, **kwargs):
    """Make data loaders with different datasets.

    Args:
        cfg (Config object): pass in all configurations.
        modes (list of str): list of data loader modes to be executed.
        kwargs: passed to DataLoader.
    """
    data_loaders = []

    for mode in modes:
        data_set = InstSeg(cfg=cfg, mode=mode)
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
