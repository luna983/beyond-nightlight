import numpy as np
import argparse

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

from utils.configure import Config
from utils.save_checkpoint import Saver
from utils.log_tensorboard import Writer
from dataloader import make_data_loader

class Trainer(object):
    """Train and evaluate the machine learning model.

    Args:
        cfg (Config object): stores all configurations.
    """
    def __init__(self, cfg):

        print('=' * 72)
        print("Initalizing trainer.")

        # save configurations
        self.cfg = cfg

        print("Set up logging.")

        # initialize saver and output config
        self.saver = Saver(cfg)
        self.saver.save_config()
        
        # tensorboard summary
        self.writer = Writer(cfg)

        print("Initalizing data loader.")

        # set device, detect cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # make data loader
        params = {
            'batch_size': (cfg.batch_size_per_gpu if cfg.num_gpus == 0
                else cfg.batch_size_per_gpu * cfg.num_gpus),
            'num_workers': cfg.num_workers}
        self.train_loader, self.val_loader = make_data_loader(cfg, **params)

        print("Initalizing model and optimizer.")

        # make model
        params = {
            'num_classes': len(cfg.label_dict) + 1, # including background
            'pretrained': cfg.coco_pretrained}
        self.model = maskrcnn_resnet50_fpn(**params)
        self.model.to(self.device)

        # make optimizer
        params = {
            'lr': cfg.lr}
        self.optimizer = torch.optim.Adam(self.model.parameters(), **params)

        print("Starting from epoch {} to epoch {}".format(0, self.cfg.epochs))

    def train(self, epoch):
        """Train the model.

        Args:
            epoch (int): number of epochs since training started. (starts with 0)
        """
        print('=' * 72)
        print("Epoch [{} / {}]".format(epoch, self.cfg.epochs))
        self.model.train()
        for i, sample in enumerate(self.train_loader):
            images, targets = sample
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            loss = sum([l for l in loss_dict.values()])
            print("Iteration [{}]: loss: {:.4f}".format(i, loss))
            print("; ".join(["{}: {:.4f}".format(k, v) for k, v in loss_dict.items()]) + ".")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, epoch):
        """Validate the model.

        Args:
            epoch (int): number of epochs since training started. (starts with 0)
        """
        pass

    def close(self, epoch):
        """Properly finish training.

        Args:
            epoch (int): current epoch
        """
        self.writer.close()
        print('=' * 72)
        print("Training finished, completed {} epochs.".format(epoch))
        print('=' * 72)

if __name__ == '__main__':

    # collect command line arguments
    parser = argparse.ArgumentParser(description="Visualize instance segmentation masks.")
    parser.add_argument('--config', nargs='+', type=str,
                        default=["default_config.yaml"],
                        help="Path to config files.")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Do not use CUDA.")
    parser.add_argument('--cuda-max-devices', type=int, default=1,
                        help="Maximum number of available GPUs.")
    args = parser.parse_args()

    # parse configurations
    cfg = Config()
    cfg.update(args.config)

    # check CUDA
    if not args.no_cuda:
        assert torch.cuda.is_available(), "CUDA not available."
    if args.cuda_max_devices is not None:
        cfg.num_gpus = torch.cuda.device_count()
        assert cfg.num_gpus <= args.cuda_max_devices, (
            "{} GPUs available, please set visible devices.\n".format(cfg.num_gpus) +
            "export CUDA_VISIBLE_DEVICES=X,X")

    # train
    trainer = Trainer(cfg)
    for epoch in range(0, trainer.cfg.epochs):
        trainer.train(epoch)
        trainer.validate(epoch)
    trainer.close(trainer.cfg.epochs)
