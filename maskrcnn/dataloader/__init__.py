from torch.utils.data import DataLoader

from .inst_seg import InstSeg


def collate_fn_drop_empty(batch):
    """Helper function to collate lists of image/target pairs into a batch.

    Args:
        batch (tuple): image target pairs.

    Returns:
        tuple: containing N images and N targets, dropping empty images
    """
    nonempty = [(image, target)
                for image, target in batch if target is not None]
    return tuple(zip(*nonempty))


def collate_fn(batch):
    """Helper function to collate lists of image/target pairs into a batch.

    Args:
        batch (tuple): image target pairs.

    Returns:
        tuple: containing N images and N targets
    """
    return tuple(zip(*batch))


def make_data_loader(cfg, modes, **kwargs):
    """Make data loaders with different datasets.

    Args:
        cfg (Config object): pass in all configurations.
        modes (list of str): list of data loader modes to be executed.
        **kwargs: passed to DataLoader.

    Returns
        list of torch.utils.data.DataLoader: list of data loaders
        list of str: list of image_id
    """
    data_loaders = []
    ids = []

    for mode in modes:
        data_set = InstSeg(cfg=cfg, mode=mode)
        # shuffle data for training set but not validation and inference
        if mode in ['train']:
            data_loader = DataLoader(
                data_set, shuffle=True,
                collate_fn=collate_fn_drop_empty, pin_memory=True,
                **kwargs)
            ids.append(DataLoader([''] * len(data_set), **kwargs))
        elif mode in ['val']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_fn_drop_empty, pin_memory=True,
                **kwargs)
            ids.append(DataLoader(data_set.ids, **kwargs))
        elif mode in ['infer']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_fn, pin_memory=True,
                **kwargs)
            ids.append(DataLoader(data_set.ids, **kwargs))
        else:
            raise NotImplementedError

        data_loaders.append(data_loader)

    return data_loaders, ids
