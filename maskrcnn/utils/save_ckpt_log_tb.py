import os
import numpy as np
import json
import yaml
import shutil
from glob import glob
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from .visualize_mask import InstSegVisualization


class Saver(object):
    """Saves the models and configs.

    Args:
        cfg (Config object): stores all configurations.
    """

    def __init__(self, cfg):
        # check and create directories
        self.runs_dir = cfg.runs_dir
        if cfg.resume_dir is None:
            runs = sorted(glob(os.path.join(self.runs_dir, 'run_*')))
            run_id = int(runs[-1].split('_')[1]) + 1 if runs else 0
            self.run_dir = os.path.join(
                self.runs_dir, 'run_{:02d}_{}'.format(run_id, cfg.comment))
            os.mkdir(self.run_dir)
        else:
            self.run_dir = cfg.resume_dir
        cfg.run_dir = self.run_dir
        # save all configurations
        self.cfg = cfg

    def save_config(self):
        """Saves the config files."""
        with open(os.path.join(self.run_dir, 'config.yaml'), 'a') as f:
            yaml.dump(self.cfg, f)

    def save_checkpoint(self, state_dict, metrics=None, is_best=False):
        """Saves the checkpoint.

        Args:
            state_dict (dict): a dict storing all relevant parameters,
                including epoch, state_dict of models and optimizers.
            metrics (dict): a dict of evaluation metrics.
            is_best (bool): overwrites the best checkpoints.
        """
        # save current checkpoint
        torch.save(state_dict,
                   os.path.join(self.run_dir, 'checkpoint.pth.tar'))
        if metrics is not None:
            with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f)
        if is_best:
            shutil.copyfile(
                os.path.join(self.run_dir, 'checkpoint.pth.tar'),
                os.path.join(self.run_dir, 'best_checkpoint.pth.tar'))
            if metrics is not None:
                shutil.copyfile(
                    os.path.join(self.run_dir, 'metrics.json'),
                    os.path.join(self.run_dir, 'best_metrics.json'))

    def create_tb_summary(self):
        self.writer = SummaryWriter(log_dir=self.run_dir)

    def log_tb_loss(self, mode, losses, loss_dicts, epoch):
        """Log loss on Tensorboard.

        Args:
            mode (str): mode should be in ['train'].
            losses (list of torch.Tensor): the losses from this epoch.
            loss_dicts (list of dict): dicts of different types of losses, they
                sum up to total loss.
            epoch (int): number of epochs.
        """
        loss = sum([l for l in losses]) / len(losses)
        loss_dict = defaultdict(list)
        for l in loss_dicts:
            for k, v in l.items():
                loss_dict[k].append(v)
        self.writer.add_scalar('/'.join((mode, 'loss')), loss, epoch)
        for k, vs in loss_dict.items():
            loss_component = sum([v for v in vs]) / len(vs)
            self.writer.add_scalar('/'.join((mode, k)), loss_component, epoch)

    def log_tb_eval(self, mode, metrics, epoch):
        """Log evaluation on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val'].
            metrics (dict): dict of evaluation metrics.
            epoch (int): number of epochs.
        """
        if metrics is not None:
            for k, v in metrics.items():
                if np.isscalar(v):
                    self.writer.add_scalar('/'.join((mode, k)), v, epoch)

    def log_tb_visualization(self, mode, epoch,
                             image, target, pred,
                             file_name=None):
        """Log visualization on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val', 'infer'].
            epoch (int): number of epochs since training started.
            image (torch.Tensor): image to be plotted.
            target (dict of torch.Tensor): a dict of torch tensors
                following Mask RCNN conventions. Ground truth.
            pred (dict of torch.Tensor): a dict of torch tensors
                following Mask RCNN conventions. Predictions.
            file_name (str): name of the file saved.
        """
        if np.random.random() < self.cfg.prob_visualization:
            i = np.random.randint(self.cfg.num_visualization)
            file_name = '{:02d}'.format(i) if file_name is None else file_name
            # ground truth
            if target is not None:
                os.makedirs(
                    os.path.join(self.cfg.out_visual_dir, mode, 'gt'),
                    exist_ok=True)
                v = InstSegVisualization(
                    self.cfg, image=image,
                    boxes=target['boxes'], labels=target['labels'],
                    masks=target['masks'])
                v.plot_image()
                v.add_bbox()
                v.add_label()
                v.add_binary_mask()
                v.save(os.path.join(
                    self.cfg.out_visual_dir, mode, 'gt',
                    '{}.png'.format(file_name)))
                self.writer.add_image(
                    '/'.join((mode, 'gt_{:02d}'.format(i))),
                    np.array(v.output),
                    epoch,
                    dataformats='HWC')
            # predictions
            if pred is not None:
                os.makedirs(
                    os.path.join(self.cfg.out_visual_dir, mode, 'pred'),
                    exist_ok=True)
                v = InstSegVisualization(
                    self.cfg, image=image,
                    boxes=pred['boxes'], labels=pred['labels'],
                    scores=pred['scores'], masks=pred['masks'])
                v.plot_image()
                v.add_bbox()
                v.add_label_score()
                v.add_binary_mask()
                v.save(os.path.join(
                    self.cfg.out_visual_dir, mode, 'pred',
                    '{}.png'.format(file_name)))
                self.writer.add_image(
                    '/'.join((mode, 'pred_{:02d}'.format(i))),
                    np.array(v.output),
                    epoch,
                    dataformats='HWC')

    def close_tb_summary(self):
        self.writer.close()
