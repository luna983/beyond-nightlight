from torch.utils.data import DataLoader

from .inst_seg import InstSeg


def collate_fn_train(batch):
    """Helper function to collate lists of image/target pairs into a batch.

    Args:
        batch (tuple): image target pairs.

    Returns:
        tuple[N]: containing N (image, target) pairs, dropping empty images
            in training mode
    """
    return tuple([(image, target) for image, target in zip(*batch)
                  if target is not None])


def collate_fn_infer(batch):
    """Helper function to collate lists of image/target pairs into a batch.

    Args:
        batch (tuple): image target pairs.

    Returns:
        tuple[N]: containing N (image, target) pairs, dropping empty images
            in training mode
    """
    return tuple(zip(*batch))


def make_data_loader(cfg, modes, **kwargs):
    """Make data loaders with different datasets.

    Args:
        cfg (Config object): pass in all configurations.
        modes (list of str): list of data loader modes to be executed.
        kwargs: passed to DataLoader.

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
                collate_fn=collate_fn_train, pin_memory=True,
                **kwargs)
        elif mode in ['val', 'infer']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_fn_infer, pin_memory=True,
                **kwargs)
        else:
            raise NotImplementedError

        data_loaders.append(data_loader)
        ids.append(data_set.int_ids)

    return data_loaders, ids
