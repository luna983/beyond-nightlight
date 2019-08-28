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


class OpenAITanzaniaInstSeg(Dataset):
    """This loads the OpenAITanzania dataset.

    Args:
        cfg (Config): pass all configurations into the dataloader
        mode (str): a string in ['train', 'val', 'infer'],
            indicating the train, validation or inference mode
            for which data is loaded
        drop_empty (bool): whether empty images with no instances
            should be dropped
        init (bool): whether initialization (train/val split) should
            be implemented, defaults to False but initialization
            auto implemented when not detecting train.txt and val.txt files
        train_ratio (float): ratio of training images (as opposed to val)
    """

    def __init__(self, cfg, mode, drop_empty=True,
                 init=False, train_ratio=0.85):

        # log
        self.cfg = cfg
        self.mode = mode

        # initialize the train/val split
        f_exist = (os.path.isfile(os.path.join(cfg.in_trainval_dir,
                                               'train.txt')) and
                   os.path.isfile(os.path.join(cfg.in_trainval_dir,
                                               'val.txt')))
        if init or (mode in ['train', 'val'] and not f_exist):
            files = [f for f in glob(
                os.path.join(cfg.in_trainval_dir,
                             cfg.in_trainval_mask_dir,
                             '*.json'))]
            if drop_empty:
                nonempty_files = []
                for file in files:
                    f = InstanceMask()
                    f.from_supervisely(
                        file=file, label_dict=self.cfg.label_dict)
                    if not len(f) == 0:
                        nonempty_files.append(file)
                files = nonempty_files
            else:
                raise NotImplementedError
            ids = [os.path.basename(f).split('.')[0] for f in files]
            shuffle(ids)
            train_idx = np.round(train_ratio * len(ids)).astype(np.uint32)
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


        # self.args = args
        # self.split = split
        # self.base_dir = base_dir

        # if self.split in ['train', 'val']:
        #     with open(os.path.join(base_dir, split + '.txt'), 'r') as f:
        #         self.image_ids = json.load(f)
        #     self.images = [os.path.join(base_dir, 'ChipImage', image_id + '.png') for image_id in self.image_ids] 
        #     self.categories = [os.path.join(base_dir, 'ChipAnn', image_id + '.png') for image_id in self.image_ids]
        #     assert len(self.images) == len(self.categories)
        #     # Display stats
        #     print("Number of images in {}: {:d}".format(split, len(self.images)))
        # else:
        #     raise NotImplementedError
        

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return "Open AI Tanzania {} sample".format(self.split)

    def __getitem__(self, index):
        if self.split in ['train', 'val']:
            image = Image.open(self.images[index])
            label = Image.open(self.categories[index])
            sample = {'image': image, 'label': label}
            if self.split == 'train':
                return self.transform_train(sample)
            elif self.split == 'val':
                return self.transform_val(sample)
        else:
            raise NotImplementedError

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=513),
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=513),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)


if __name__ == '__main__':

    base_dir = "../../../data/OpenAITanzania/"
    train_ratio = 0.85

    files = [os.path.basename(file_dir).split(".")[0]
             for file_dir in glob(os.path.join(base_dir, 'ChipAnn', '*.png'))]
    shuffle(files)

    train_idx = np.round(train_ratio * len(files)).astype(np.int)
    with open(os.path.join(base_dir, 'train.txt'), 'w') as output_file:
        json.dump(files[0:train_idx], output_file)
    with open(os.path.join(base_dir, 'val.txt'), 'w') as output_file:
        json.dump(files[train_idx:len(files)], output_file)

    # test data loader
    from dataloaders import make_data_loader
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'openaitanzania'
    args.data_dir = base_dir
    args.num_class = 4
    args.batch_size = 12

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    for i, sample in enumerate(train_loader):
        print(i, sample['image'].size(), sample['label'].size())
        break

    for i, sample in enumerate(val_loader):
        print(i, sample['image'].size(), sample['label'].size())
        break
