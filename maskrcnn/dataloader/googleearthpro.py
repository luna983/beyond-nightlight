import os
from glob import glob
from random import shuffle
from PIL import Image

import torchvision
from torch.utils.data import Dataset

import transforms as tr
from mask_transforms import InstanceMask

# import json
# import numpy as np
# import csv



class GoogleEarthProInstSeg(Dataset):
    """This loads the Google Earth Pro dataset.

    Args:
        cfg (Config): pass all configurations into the dataloader
        mode (str): a string in ['train', 'val', 'infer'], indicating the train,
            validation or inference mode for which data is loaded
        init (bool): whether initialization (train/val split) should be implemented,
            defaults to False but initialization auto implemented when not detecting
            train.txt and val.txt files
    """
    
    def __init__(self, cfg, mode, init=False, train_ratio=0.85):

        super().__init__()

        assert mode in ['train', 'val', 'infer']

        # initialize the train/val split
        if init or (mode in ['train', 'val'] and not (
            os.path.isfile(os.path.join(cfg.in_trainval_dir, 'train.txt')) and
            os.path.isfile(os.path.join(cfg.in_trainval_dir, 'val.txt')))):
            ids = [os.path.basename(f).split('.')[0] for f in glob(
                os.path.join(cfg.in_trainval_dir, cfg.in_trainval_img_dir, '*.jpg'))]
            shuffle(ids)
            train_idx = np.round(train_ratio * len(ids)).astype(np.int)
            with open(os.path.join(cfg.in_trainval_dir, 'train.txt'), 'w') as f:
                json.dump(ids[:train_idx], f)
            with open(os.path.join(cfg.in_trainval_dir, 'val.txt'), 'w') as f:
                json.dump(ids[train_idx:], f)

        # load train/val/inference data
        if mode in ['train', 'val']:
            with open(os.path.join(cfg.in_trainval_dir, mode + '.txt'), 'r') as f:
                self.ids = json.load(f)
            self.images = [os.path.join(
                cfg.in_trainval_dir, cfg.in_trainval_img_dir, i + '.jpg') for i in self.ids] 
            self.targets = [os.path.join(
                cfg.in_trainval_dir, cfg.in_trainval_mask_dir, i + '.jpg.json') for i in self.ids]      
        elif mode in ['infer']:
            self.images = glob(os.path.join(cfg.in_infer_dir, '*.jpg'))
            self.ids = [os.path.basename(f).split('.')[0] for f in self.images]
            self.targets = []
        else:
            raise NotImplementedError

        # log 
        self.cfg = cfg
        self.mode = mode

        # check file existence
        assert all([os.path.isfile(f) for f in self.images])
        assert all([os.path.isfile(f) for f in self.targets])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        
        if self.mode in ['train', 'val']:
            target = InstanceMask().from_supervisely(
                file=self.targets[index], label_dict=cfg.label_dict)
            sample = {'image': image, 'target': target}
            if self.mode == 'train':
                return self.transform_train(sample)
            elif self.mode == 'val':
                return self.transform_val(sample)
        elif self.mode in ['infer']:
            return self.transform_infer(image)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return "Google Earth Pro {} sample: {:d} images".format(self.mode, len(self.images))

    def transform_train(self, sample):
        composed_transforms = tr.Compose([
            tr.ToTensor(),
            tr.Normalize()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = tr.Compose([
            tr.ToTensor(),
            tr.Normalize()])
        return composed_transforms(sample)

    def transform_infer(self, image):
        composed_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return composed_transforms(image)