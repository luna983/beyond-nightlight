import torch
import warnings

from torch.utils.data import DataLoader

from .googleearthpro import GoogleEarthProInstSeg

if torch.__version__ < '1.1.0':
    warnings.warn("PyTorch version below 1.1.0. pin_memory option not available.")

if not torch.cuda.is_available():
    warnings.warn("CUDA is not available.")

class MaskRCNNBatch(object):
    """Define methods to collate a list of data into a mini-batch.

    Args:
        data (list of dataset (subclass of torch.utils.data.Dataset)
            items (from __getitem__)): data to be formed into a mini-batch.
        has_target (bool): whether the batch has a target object
            (True for train, val, False for infer)
    """

    def __init__(self, data, has_target=True):
        if has_target:
            self.image, self.target = zip(*data)
            self.image = list(self.image)
            self.target = list(self.target)
        else:
            self.image = data

    def pin_memory(self):
        for tensor in self.image:
            tensor = tensor.pin_memory()
        if hasattr(self, target):
            for item in self.target:
                item['masks'] = item['masks'].pin_memory()
                item['labels'] = item['labels'].pin_memory()
                item['boxes'] = item['boxes'].pin_memory()
        return self

def collate_maskrcnn_image_target(batch):
    return MaskRCNNBatch(batch, has_target=True)

def collate_maskrcnn_image(batch):
    return MaskRCNNBatch(batch, has_target=False)

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
            data_loader = DataLoader(
                data_set, shuffle=True,
                collate_fn=collate_maskrcnn_image_target, pin_memory=True,
                **kwargs)
        elif mode in ['val']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_maskrcnn_image_target, pin_memory=True,
                **kwargs)
        elif mode in ['infer']:
            data_loader = DataLoader(
                data_set, shuffle=False,
                collate_fn=collate_maskrcnn_image, pin_memory=True,
                **kwargs)
        else:
            raise NotImplementedError

        data_loaders.append(data_loader)

    return data_loaders