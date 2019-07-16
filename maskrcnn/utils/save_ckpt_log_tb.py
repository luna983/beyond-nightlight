import os
import numpy as np
import json
import yaml
import shutil
from glob import glob
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.visualize_mask import InstSegVisualization


class Saver(object):
    """Saves the models and configs.

    Args:
        cfg (Config object): stores all configurations.
    """
    def __init__(self, cfg):
        # check and create directories
        self.runs_dir = cfg.runs_dir
        if cfg.resume_dir is None:
            runs = sorted(glob(os.path.join(self.runs_dir, "run_*")))
            run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
            self.run_dir = os.path.join(self.runs_dir, "run_{:02d}".format(run_id))
            if not os.path.exists(self.run_dir):
                os.mkdir(self.run_dir)
        else:
            self.run_dir = cfg.resume_dir
        cfg.run_dir = self.run_dir
        # save all configurations
        self.cfg = cfg

    def save_config(self):
        """Saves the config files."""
        with open(os.path.join(self.run_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.cfg, f)

    def save_checkpoint(self, state_dict, save_best=True,
                        metrics=None, key_metric_name=None, best_metrics=None):
        """Saves the checkpoint.

        Args:
            state_dict (dict): a dict storing all relevant parameters, including
                epoch, state_dict of models and optimizers.
            save_best (bool): overwrites the best checkpoints.
            metrics (dict): a dict of evaluation metrics.
            key_metric_name (str): name of the key metric to determine the best model.
                Required if save_best=True.
            best_metrics (dict): the best key metrics for prior models.
                Required if save_best=True.
        """
        # save current checkpoint
        torch.save(state_dict, os.path.join(self.run_dir, "checkpoint.pth.tar"))
        if metrics is not None:
            with open(os.path.join(self.run_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f)

        if save_best:
            assert key_metric_name is not None, "No metric provided for comparison."
            if best_metrics is None:
                is_best = True
            else:
                if metrics[key_metric_name] > best_metrics[key_metric_name]:
                    is_best = True
                else:
                    is_best = False
            if is_best:
                shutil.copyfile(
                    os.path.join(self.run_dir, "checkpoint.pth.tar"),
                    os.path.join(self.run_dir, "best.pth.tar"))
                if metrics is not None:
                    shutil.copyfile(
                        os.path.join(self.run_dir, "metrics.json"),
                        os.path.join(self.run_dir, "best.json"))

    def create_tb_summary(self):
        self.writer = SummaryWriter(log_dir=self.run_dir)

    def log_tb_loss(self, mode, losses, loss_dicts, epoch):
        """Log loss on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val'].
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
            l = sum([v for v in vs]) / len(vs)
            self.writer.add_scalar('/'.join((mode, k)), l, epoch)

    def log_tb_eval(self, mode, metrics, epoch):
        """Log evaluation on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val'].
            metrics (dict): dict of evaluation metrics.
            epoch (int): number of epochs.
        """
        if metrics is not None:
            for k, v in metrics.items():
                self.writer.add_scalar('/'.join((mode, k)), v, epoch)

    def log_tb_visualization(self, mode, epoch, image, target, pred):
        """Log visualization on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val'].
            epoch (int): number of epochs since training started.
            image (torch.Tensor): image to be plotted.
            target (dict of torch.Tensor): a dict of torch tensors 
                following Mask RCNN conventions. Ground truth.
            pred (dict of torch.Tensor): a dict of torch tensors 
                following Mask RCNN conventions. Predictions.
        """
        if np.random.random() < self.cfg.prob_train_visualization:
            v = InstSegVisualization(
                self.cfg, image=image,
                boxes=target['boxes'], labels=target['labels'],
                masks=target['masks'])
            v.plot_image()
            v.add_bbox()
            v.add_label()
            # v.add_binary_mask()
            v.save(os.path.join(self.run_dir, "visualization_gt.png"))
            self.writer.add_image(
                '/'.join((mode, 'ground_truth')),
                np.array(v.output),
                epoch,
                dataformats='HWC')
            # predictions
            v = InstSegVisualization(
                self.cfg, image=image,
                boxes=pred['boxes'], labels=pred['labels'],
                scores=pred['scores'], masks=pred['masks'])
            v.plot_image()
            v.add_bbox()
            v.add_label_score()
            # v.add_binary_mask()
            v.save(os.path.join(self.run_dir, "visualization_pred.png"))
            self.writer.add_image(
                '/'.join((mode, 'predictions')),
                np.array(v.output),
                epoch,
                dataformats='HWC')

    def close_tb_summary(self):
        self.writer.close()
