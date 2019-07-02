import os
import numpy as np
from glob import glob
import argparse

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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
        print("Initalizing trainer...")
        # initialize saver and output config
        self.saver = Saver(cfg)
        self.saver.save_config()
        # tensorboard summary
        self.saver.create_tb_summary()

        print("Initalizing data loader...")

        # set device, detect cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # make data loader
        params = {
            'batch_size': (cfg.batch_size_per_gpu if cfg.num_gpus == 0
                else cfg.batch_size_per_gpu * cfg.num_gpus),
            'num_workers': cfg.num_workers}
        self.train_loader, self.val_loader = make_data_loader(cfg, **params)

        print("Initalizing model and optimizer...")

        # make model
        if cfg.coco_pretrained:
            self.model = maskrcnn_resnet50_fpn(pretrained=True)
            # replace the pre-trained FasterRCNN head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features=self.model.roi_heads.box_predictor.cls_score.in_features,
                num_classes=len(cfg.label_dict) + 1)
            # replace the pre-trained MaskRCNN head with a new one
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask=self.model.roi_heads.mask_predictor.conv5_mask.in_channels,
                hidden_layer=self.model.roi_heads.mask_predictor.conv5_mask.dim_reduced,
                num_classes=len(cfg.label_dict) + 1)
        else:
            params = {
                'num_classes': len(cfg.label_dict) + 1, # including background
                'pretrained': False}
            self.model = maskrcnn_resnet50_fpn(**params)
        self.model.to(self.device)

        # make optimizer
        params = {
            'lr': cfg.lr}
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            **params)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_scheduler_step_size, gamma=cfg.lr_scheduler_gamma)

        # load prior checkpoint
        if cfg.run_dir is not None:
            ckpt_file = os.path.join(cfg.run_dir, "checkpoint.pth.tar")
            key_metric_file = os.path.join(
                cfg.run_dir, "best_" + cfg.key_metric_name + ".txt")
            assert os.path.isfile(key_metric_file)
            assert os.path.isfile(ckpt_file)
            print("Loading checkpoint {}".format(ckpt_file)
            ckpt = torch.load(ckpt_file)
            self.start_epoch = ckpt['epoch']
            if torch.cuda.is_available():
                self.model.module.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            with open(key_metric_file, 'r') as f:
                self.best_key_metric = float(f.readline())
        else:
            self.start_epoch = 0
            self.best_key_metric = cfg.key_metric_init

        print("Starting from epoch {} to epoch {}...".format(self.start_epoch, cfg.epochs))
        # save configurations
        self.cfg = cfg

    def train(self, epoch):
        """Train the model.

        Args:
            epoch (int): number of epochs since training started. (starts with 0)
        """
        print('=' * 72)
        print("Epoch [{} / {}]".format(epoch, self.cfg.epochs))
        print("Training...")
        self.model.train()
        for i, sample in enumerate(self.train_loader):
            images, targets = sample
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            # forward pass
            loss_dict = self.model(images, targets)
            loss = sum([l for l in loss_dict.values()])
            self.saver.log_tb_loss(mode='train', loss=loss, loss_dict=loss_dict, epoch=epoch)
            print("Iteration [{}]: loss: {:.4f}".format(i, loss))
            print('; '.join(["{}: {:.4f}".format(k, v) for k, v in loss_dict.items()]) + '.')
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update the learning rate
            self.lr_scheduler.step()

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model.

        Args:
            epoch (int): number of epochs since training started. (starts with 0)
        """
        print("Validating...")
        self.model.eval()
        for i, sample in enumerate(self.val_loader):
            images, targets = sample
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            loss = sum([l for l in loss_dict.values()])
            self.saver.log_tb_loss(mode='val', loss=loss, loss_dict=loss_dict, epoch=epoch)
            print("Iteration [{}]: loss: {:.4f}".format(i, loss))
            print('; '.join(["{}: {:.4f}".format(k, v) for k, v in loss_dict.items()]) + '.')
        # save checkpoint every epoch
        self.saver.save_checkpoint(
            state_dict={'epoch': epoch,
                        'state_dict': (self.model.module.state_dict() if torch.cuda.is_available()
                            else self.model.state_dict()),
                        'optimizer': self.optimizer.state_dict()},
            save_best=False)

    def close(self, epoch):
        """Properly finish training.

        Args:
            epoch (int): current epoch
        """
        self.saver.close_tb_summary()
        print('=' * 72)
        print("Training finished, completed {} to {} epochs.".format(
            self.start_epoch, epoch))
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
    parser.add_argument('--resume-run', type=str, default=None,
                        help="Load existing checkpoint and resume training.")
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
    assert os.path.exists(cfg.runs_dir), "Model/log directory does not exist."
    if args.resume_run is not None:
        assert os.path.exists(os.path.join(cfg.runs_dir, args.resume_run))
        cfg.run_dir = os.path.join(cfg.runs_dir, args.resume_run)
    else
        cfg.run_dir = None

    # train
    try:
        trainer = Trainer(cfg)
        for epoch in range(0, trainer.cfg.epochs):
            trainer.train(epoch)
            trainer.validate(epoch)
    finally:
        trainer.close(epoch)
