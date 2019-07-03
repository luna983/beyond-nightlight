import os
import yaml
import shutil
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter

class Saver(object):
    """Saves the models and configs.

    Args:
        cfg (Config object): stores all configurations.
    """
    def __init__(self, cfg):
        # check and create directories
        self.runs_dir = cfg.runs_dir
        if cfg.run_dir is None:
            runs = sorted(glob(os.path.join(self.runs_dir, "run_*")))
            run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
            self.run_dir = os.path.join(self.runs_dir, "run_{:02d}".format(run_id))
            if not os.path.exists(self.run_dir):
                os.mkdir(self.run_dir)
        else:
            self.run_dir = cfg.run_dir
        # save all configurations
        self.cfg = cfg

    def save_config(self):
        """Saves the config files."""
        with open(os.path.join(self.run_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.cfg)

    def save_checkpoint(self, state_dict,
                        save_best=True, key_metric=None, best_key_metric=None):
        """Saves the checkpoint.

        Args:
            state_dict (dict): a dict storing all relevant parameters, including
                epoch, state_dict of models and optimizers, and key metrics.
            save_best (bool): overwrites the best checkpoints.
            key_metric (float): a key metric to determine the best model.
                Required if save_best=True.
            best_key_metric (float): the best key metric to determine the best model.
                Required if save_best=True.
        """
        # save current checkpoint
        torch.save(state_dict, os.path.join(self.run_dir, "checkpoint.pth.tar"))

        if save_best:
            assert key_metric is not None, "No metric provided for comparison."
            if key_metric > best_key_metric:
                shutil.copyfile(
                    os.path.join(self.run_dir, "checkpoint.pth.tar"),
                    os.path.join(self.run_dir, "best.pth.tar"))
                key_metric_file = os.path.join(
                    self.run_dir, "best_" + self.cfg.key_metric_name + ".txt")
                with open(key_metric_file, 'w') as f:
                    f.write(str(key_metric))

    def create_tb_summary(self):
        self.writer = SummaryWriter(log_dir=self.run_dir)

    def log_tb_loss(self, mode, loss, loss_dict, epoch):
        """Log loss on Tensorboard.

        Args:
            mode (str): mode should be in ['train', 'val'].
            loss (torch.Tensor): the loss from this epoch.
            loss_dict (dict): dict of different types of losses, they
                sum up to total loss.
            epoch (int): number of epochs.
        """
        self.writer.add_scalar('/'.join((mode, 'loss')), loss, epoch)
        for k, v in loss_dict.items():
            self.writer.add_scalar('/'.join((mode, k)), v, epoch)

    def log_tb_visualization(self):
        pass

    def close_tb_summary(self):
        self.writer.close()
