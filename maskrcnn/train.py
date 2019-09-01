import os
import json
import argparse
import copy
from tqdm import tqdm

import torch

from dataloader import make_data_loader
from model import make_model
from utils.configure import Config
from utils.save_ckpt_log_tb import Saver
from utils.coco import COCOSaver
from utils.eval import evaluate


class Trainer(object):
    """Train and evaluate the machine learning model.

    Args:
        cfg (Config object): stores all configurations.
    """

    def __init__(self, cfg):

        print('=' * 72)
        print('Initalizing trainer...')
        # initialize saver and output config
        self.saver = Saver(cfg)
        self.saver.save_config()
        # tensorboard summary
        self.saver.create_tb_summary()

        print('Initalizing data loader...')

        # set device, detect cuda availability
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

        # make data loader
        cfg.batch_size = (cfg.batch_size_per_gpu if cfg.num_gpus == 0
                          else cfg.batch_size_per_gpu * cfg.num_gpus)
        params = {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers}
        if cfg.infer:
            self.val_loader = make_data_loader(
                cfg, modes=['infer'], **params)[0]
        else:
            self.train_loader, self.val_loader = make_data_loader(
                cfg, **params)
            # TODO: comment this out later
            self.val_loader = self.train_loader

        print('Initalizing model and optimizer...')
        cfg.num_classes = len(cfg.label_dict) + 1  # including background
        self.model = make_model(cfg)
        self.model.to(self.device)

        # make optimizer
        params = {
            'lr': cfg.lr}
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            **params)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_scheduler_step_size,
            gamma=cfg.lr_scheduler_gamma)

        # load prior checkpoint
        ckpt_file = os.path.join(cfg.run_dir, 'checkpoint.pth.tar')
        if os.path.isfile(ckpt_file):
            print('Loading checkpoint {}'.format(ckpt_file))
            ckpt = torch.load(ckpt_file)
            self.start_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            self.start_epoch = 0
        # load prior stats
        metrics_file = os.path.join(cfg.run_dir, 'best_metrics.json')
        if os.path.isfile(metrics_file):
            with open(metrics_file, 'r') as f:
                self.best_metrics = json.load(f)
        else:
            self.best_metrics = None
        if not cfg.infer:
            print('Starting from epoch {} to epoch {}...'
                  .format(self.start_epoch, cfg.epochs - 1))
        # save configurations
        self.cfg = cfg

    def train(self, epoch):
        """Train the model.

        Args:
            epoch (int): number of epochs since training started
        """

        print('=' * 72)
        print('Epoch [{} / {}]'.format(epoch, self.cfg.epochs - 1))
        print('Training...')
        self.model.train()
        losses = []
        loss_dicts = []
        for i, sample in enumerate(self.train_loader):
            images, targets = sample
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]
            # forward pass
            loss_dict = self.model(images, targets)
            loss = sum([l for l in loss_dict.values()])
            print('Iteration [{}]: loss: {:.4f}'.format(i, loss))
            print('; '.join(['{}: {:.4f}'.format(k, v)
                             for k, v in loss_dict.items()]) + '.')
            losses.append(loss.detach().cpu())
            loss_dicts.append({k: v.detach().cpu()
                               for k, v in loss_dict.items()})
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update the learning rate
            self.lr_scheduler.step()
        self.saver.log_tb_loss(mode='train', losses=losses,
                               loss_dicts=loss_dicts, epoch=epoch)

    def save_val_annotations(self):
        """Saves the validation set annotations as COCO format.
        """
        cocosaver = COCOSaver(gt=True, cfg=self.cfg)
        for i, sample in enumerate(self.val_loader):
            _, targets = sample
            for target in targets:
                cocosaver.add(target)
        cocosaver.save()

    @torch.no_grad()
    def infer(self, epoch):
        """Run inference on the model.

        Args:
            epoch (int): number of epochs since training started.
        """
        print('Running inference...')
        cocosaver = COCOSaver(gt=False, cfg=self.cfg)
        self.model.eval()
        for sample in tqdm(self.val_loader):
            images, targets = sample
            images_copy = copy.deepcopy(images)
            images = [im.to(self.device) for im in images]
            preds = self.model(images)
            for image, target, pred in zip(images_copy, targets, preds):
                pred['masks'] = (pred['masks'].squeeze(1) >
                                 self.cfg.mask_threshold)
                cocosaver.add(pred)
                self.saver.log_tb_visualization(
                    mode=('infer' if self.cfg.infer else 'val'),
                    epoch=epoch,
                    image=image,
                    target=target,
                    pred=pred)
        cocosaver.save()

    def evaluate(self, epoch):
        """Evaluates the saved predicted annotations versus ground truth.

        Args:
            epoch (int): number of epochs since training started.
        """
        self.metrics = evaluate(self.cfg)
        self.saver.log_tb_eval(mode='val', metrics=self.metrics, epoch=epoch)

    def save_checkpoint(self, epoch):
        """Saves the checkpoint.

        Args:
            epoch (int): number of epochs since training started.
        """
        # save checkpoint every epoch
        self.saver.save_checkpoint(
            state_dict={'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
            save_best=True,
            metrics=self.metrics,
            key_metric_name=self.cfg.key_metric_name,
            best_metrics=self.best_metrics)

    def close(self, epoch):
        """Properly finish training.

        Args:
            epoch (int): current epoch
        """
        self.saver.close_tb_summary()
        print('=' * 72)
        if not self.cfg.infer:
            print('Training finished, completed {} to {} epochs.'.format(
                self.start_epoch, epoch))
        else:
            print('Inference completed!')
        print('=' * 72)


if __name__ == '__main__':

    assert torch.__version__ >= '1.1.0'

    # collect command line arguments
    parser = argparse.ArgumentParser(description='Run Mask RCNN.')
    parser.add_argument('--config', nargs='+', type=str, default=None,
                        help='Path to config files.')
    parser.add_argument('--mode', nargs='+', type=str, default=None,
                        help='In train, val or infer mode.')
    parser.add_argument('--resume-run', type=str, default=None,
                        help='Load existing checkpoint and resume.')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Do not use CUDA.')
    parser.add_argument('--cuda-max-devices', type=int, default=1,
                        help='Maximum number of available GPUs.')
    args = parser.parse_args()

    # parse configurations
    cfg = Config()
    cfg.update(args.config)

    # check CUDA
    if not args.no_cuda:
        assert torch.cuda.is_available(), 'CUDA not available.'
    if args.cuda_max_devices is not None:
        cfg.num_gpus = torch.cuda.device_count()
        assert cfg.num_gpus <= args.cuda_max_devices, (
            '{} GPUs available, please set visible devices.\n'
            .format(cfg.num_gpus) +
            'export CUDA_VISIBLE_DEVICES=X')
    assert os.path.exists(cfg.runs_dir), 'Model/log directory does not exist.'
    if args.resume_run is not None:
        assert os.path.exists(os.path.join(cfg.runs_dir, args.resume_run))
        cfg.resume_dir = os.path.join(cfg.runs_dir, args.resume_run)
    else:
        cfg.resume_dir = None

    # check for inference flag
    cfg.infer = args.infer
    if not args.infer:
        # train
        trainer = Trainer(cfg)
        trainer.save_val_annotations()
        for epoch in range(trainer.start_epoch, cfg.epochs):
            trainer.train(epoch)
            if epoch >= cfg.eval_epoch:
                trainer.infer(epoch)
                trainer.evaluate(epoch)
                trainer.save_checkpoint(epoch)
        trainer.close(epoch)
    else:
        epoch = cfg.epochs + 1
        trainer = Trainer(cfg)
        trainer.infer(epoch)
        trainer.close(epoch)
