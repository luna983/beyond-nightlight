import os
import json
import numpy as np
from glob import glob
from random import shuffle
from PIL import Image

import torchvision
from torch.utils.data import Dataset

from .transforms import Compose, ToTensor, RandomCrop
from .transforms import Resize, ColorJitter
from .transforms import RandomHorizontalFlip, RandomVerticalFlip
from .mask_transforms import InstanceMask


class InstSeg(Dataset):
    """This loads any instance segmentation dataset.

    Args:
        cfg (Config): pass all configurations into the dataloader
        mode (str): a string in ['train', 'val', 'infer'],
            indicating the train, validation or inference mode
            for which data is loaded
    """

    def __init__(self, cfg, mode):
        super().__init__()
        # initialize the train/val split
        f_exist = (os.path.isfile(os.path.join(cfg.in_trainval_dir,
                                               'train.txt')) and
                   os.path.isfile(os.path.join(cfg.in_trainval_dir,
                                               'val.txt')))
        if cfg.train_val_init or (mode in ['train', 'val'] and not f_exist):
            files = [f for f in glob(
                os.path.join(cfg.in_trainval_dir,
                             cfg.in_trainval_mask_dir,
                             '*.{}'.format(cfg.mask_suffix)))]
            if cfg.drop_empty:
                nonempty_files = []
                for file in files:
                    f = InstanceMask()
                    f.from_coordinates(
                        file=file, label_dict=cfg.label_dict)
                    if not len(f) == 0:
                        nonempty_files.append(file)
                files = nonempty_files
            else:
                raise NotImplementedError
            ids = [os.path.basename(f).split('.')[0] for f in files]
            shuffle(ids)
            train_idx = np.round(cfg.train_ratio * len(ids)).astype(np.uint32)
            with open(os.path.join(cfg.in_trainval_dir,
                                   'train.txt'), 'w') as f:
                json.dump(ids[:train_idx], f)
            with open(os.path.join(cfg.in_trainval_dir, 'val.txt'), 'w') as f:
                json.dump(ids[train_idx:], f)

        # load train/val/inference data
        if mode in ['train', 'val']:
            with open(os.path.join(cfg.in_trainval_dir,
                                   mode + '.txt'), 'r') as f:
                self.ids = json.load(f)
            self.images = [os.path.join(cfg.in_trainval_dir,
                                        cfg.in_trainval_img_dir,
                                        i + '.jpg') for i in self.ids]
            self.targets = [os.path.join(cfg.in_trainval_dir,
                                         cfg.in_trainval_mask_dir,
                                         i + '.jpg.json') for i in self.ids]
        elif mode in ['infer']:
            self.images = sorted(glob(os.path.join(cfg.in_infer_dir, '*.png')))
            self.ids = [os.path.basename(f).split('.')[0] for f in self.images]
            self.targets = []
        else:
            raise NotImplementedError

        # check file existence
        assert all([os.path.isfile(f) for f in self.images])
        assert all([os.path.isfile(f) for f in self.targets])

        # log
        self.dataset = cfg.dataset
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')

        if self.mode in ['train', 'val']:
            target = InstanceMask()
            target.from_supervisely(
                file=self.targets[index], label_dict=self.cfg.label_dict)
            if self.mode == 'train':
                return self.transform_train(image, target)
            elif self.mode == 'val':
                return self.transform_val(image, target)
        elif self.mode in ['infer']:
            return self.transform_infer(image, None)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return ("{}: {} sample: {:d} images"
                .format(self.dataset, self.mode, len(self.images)))

    def transform_train(self, image, target):
        h = (np.floor(image.size[1] * self.cfg.random_crop_height)
             .astype(np.uint16))
        w = (np.floor(image.size[0] * self.cfg.random_crop_width)
             .astype(np.uint16))
        composed_transforms = Compose([
            RandomCrop(size=(h, w)),
            RandomVerticalFlip(self.cfg.vertical_flip),
            RandomHorizontalFlip(self.cfg.horizontal_flip),
            ColorJitter(
                brightness=self.cfg.brightness, contrast=self.cfg.contrast,
                saturation=self.cfg.saturation, hue=self.cfg.hue),
            Resize(width=self.cfg.resize_width, height=self.cfg.resize_height),
            ToTensor()])
        return composed_transforms(image, target)

    def transform_val(self, image, target):
        composed_transforms = Compose([
            Resize(width=self.cfg.resize_width, height=self.cfg.resize_height),
            ToTensor()])
        return composed_transforms(image, target)

    def transform_infer(self, image, target):
        composed_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=(self.cfg.resize_height, self.cfg.resize_width)),
            torchvision.transforms.ToTensor()])
        return composed_transforms(image), target