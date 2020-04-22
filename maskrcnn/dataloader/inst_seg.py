import os
import json
from glob import glob
from random import shuffle
from PIL import Image

import torchvision
from torch.utils.data import Dataset

from .transforms import (Compose, ToTensor, FillEmpty,
                         Resize,  Blur, RandomCrop, ColorJitter,
                         RandomHorizontalFlip, RandomVerticalFlip)
from .mask_transforms import InstanceMask


class InstSeg(Dataset):
    """This loads an instance segmentation dataset.

    Args:
        cfg (Config): pass all configurations into the dataloader
        mode (str): a string in ['train', 'val', 'infer'],
            indicating the train, validation or inference mode
            for which data is loaded
    """

    def __init__(self, cfg, mode):

        if mode in ['train', 'val']:
            # initialize the train/val split, if necessary
            split_exist = (os.path.isfile(os.path.join(cfg.in_tv_dir,
                                                       'train.txt')) and
                           os.path.isfile(os.path.join(cfg.in_tv_dir,
                                                       'val.txt')))
            if not split_exist:
                files = [f for f in glob(
                    os.path.join(cfg.in_tv_dir, cfg.in_tv_mask_dir,
                                 '*' + cfg.in_tv_mask_suffix))]
                ids = [os.path.basename(f).split('.')[0] for f in files]
                shuffle(ids)
                train_idx = int(cfg.train_ratio * len(ids))
                assert 0 < train_idx < len(ids), 'train/val sample is empty'
                with open(os.path.join(cfg.in_tv_dir, 'train.txt'), 'w') as f:
                    json.dump(ids[:train_idx], f)
                with open(os.path.join(cfg.in_tv_dir, 'val.txt'), 'w') as f:
                    json.dump(ids[train_idx:], f)
            # load train/val data
            with open(os.path.join(cfg.in_tv_dir,
                                   mode + '.txt'), 'r') as f:
                self.ids = json.load(f)
            self.images = [os.path.join(cfg.in_tv_dir, cfg.in_tv_img_dir,
                                        i + cfg.in_tv_img_suffix)
                           for i in self.ids]
            self.targets = [os.path.join(cfg.in_tv_dir, cfg.in_tv_mask_dir,
                                         i + cfg.in_tv_mask_suffix)
                            for i in self.ids]
        elif mode in ['infer']:
            if cfg.in_infer_img_list is None:
                self.images = sorted(glob(os.path.join(
                    cfg.in_infer_dir, cfg.in_infer_img_dir,
                    '*' + cfg.in_infer_img_suffix)))
                self.ids = [os.path.basename(f).split('.')[0]
                            for f in self.images]
            else:
                with open(os.path.join(
                        cfg.in_infer_dir,
                        cfg.in_infer_img_list + '.txt'), 'r') as f:
                    self.ids = json.load(f)
                self.images = [os.path.join(
                    cfg.in_infer_dir, cfg.in_infer_img_dir,
                    i + cfg.in_infer_img_suffix) for i in self.ids]
            self.targets = []
        else:
            raise NotImplementedError

        # check file existence
        assert all([os.path.isfile(f) for f in self.images])
        assert all([os.path.isfile(f) for f in self.targets])

        # log
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.cfg.grayscale:
            image = image.convert('L').convert('RGB')
        else:
            image = image.convert('RGB')
        if self.mode in ['train', 'val']:
            target = InstanceMask()
            target.from_file(
                file=self.targets[index],
                ann_format=self.cfg.mask_format,
                label_dict=self.cfg.label_dict,
                verbose=False)
            if self.mode in ['train']:
                return self.transform_train(image, target)
            elif self.mode in ['val']:
                return self.transform_val(image, target)
        elif self.mode in ['infer']:
            return self.transform_infer(image, None)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return ('{}: {} sample: {:d} images'
                .format(self.cfg.dataset, self.mode, len(self.images)))

    def transform_train(self, image, target):
        h = int(image.size[1] * self.cfg.random_crop_height)
        w = int(image.size[0] * self.cfg.random_crop_width)
        composed_transforms = Compose([
            RandomCrop(size=(h, w)),
            RandomVerticalFlip(self.cfg.vertical_flip),
            RandomHorizontalFlip(self.cfg.horizontal_flip),
            ColorJitter(
                brightness=self.cfg.brightness, contrast=self.cfg.contrast,
                saturation=self.cfg.saturation, hue=self.cfg.hue),
            Blur(blur_prob=self.cfg.blur_prob, blur_times=self.cfg.blur_times),
            Resize(width=self.cfg.resize_width, height=self.cfg.resize_height),
            FillEmpty(self.cfg.fillempty,
                      category_int=self.cfg.num_classes - 1),
            ToTensor()])
        return composed_transforms(image, target)

    def transform_val(self, image, target):
        composed_transforms = Compose([
            Resize(width=self.cfg.resize_width, height=self.cfg.resize_height),
            FillEmpty(self.cfg.fillempty,
                      category_int=self.cfg.num_classes - 1),
            ToTensor()])
        return composed_transforms(image, target)

    def transform_infer(self, image, target):
        composed_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=(self.cfg.resize_height, self.cfg.resize_width)),
            torchvision.transforms.ToTensor()])
        return composed_transforms(image), target
