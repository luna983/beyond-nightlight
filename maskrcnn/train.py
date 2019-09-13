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

        print('Initalizing saver...')
        # initialize saver and output config
        self.saver = Saver(cfg)
        self.saver.save_config()
        # tensorboard summary
        self.saver.create_tb_summary()

        print('Initalizing data loader...')
        # make data loader
        params = {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers}
        if 'infer' in cfg.mode:
            (self.val_loader,), (self.val_ids,) = make_data_loader(
                cfg, modes=['infer'], **params)
        else:
            ((self.train_loader, self.val_loader),
             (self.train_ids, self.val_ids)) = make_data_loader(
                cfg, modes=['train', 'val'], **params)

        print('Initalizing model and optimizer...')
        # set device, detect cuda availability
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        # make model
        self.model = make_model(cfg)
        self.model.to(self.device)
        # make optimizer
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr)

        # load prior checkpoint
        ckpt_file = os.path.join(cfg.run_dir, cfg.model_to_load)
        if os.path.isfile(ckpt_file):
            print('Loading checkpoint {}'.format(ckpt_file))
            ckpt = torch.load(ckpt_file)
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            self.start_epoch = 0
        self.epoch = self.start_epoch
        # load prior stats
        metrics_file = os.path.join(cfg.run_dir, 'best_metrics.json')
        if os.path.isfile(metrics_file):
            with open(metrics_file, 'r') as f:
                self.best_metrics = json.load(f)
        else:
            self.best_metrics = None
        self.metrics = None
        self.epoch_is_best = False
        # save configurations
        self.cfg = cfg

    def train(self):
        """Train the model."""
        self.epoch += 1
        print('=' * 72)
        print('Epoch [{}]'.format(self.epoch))
        print('Training...')
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        loss_dicts = []
        for i, sample in enumerate(self.train_loader):
            if len(sample) == 0:
                continue
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
            loss.backward()
            if (i + 1) % self.cfg.batch_size_multiplier == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.saver.log_tb_loss(mode='train', losses=losses,
                               loss_dicts=loss_dicts, epoch=self.epoch)

    def save_gt_annotations(self, mode):
        """Saves the ground truth annotations as COCO format.

        Args:
            mode (str): the sample to be evaluated (train/val).
        """
        print('Saving ground truth annotations for mode {}...'.format(mode))
        cocosaver = COCOSaver(gt=True, cfg=self.cfg)
        if mode == 'train':
            loader, image_ids = self.train_loader, self.train_ids
        elif mode == 'val':
            loader, image_ids = self.val_loader, self.val_ids
        else:
            raise NotImplementedError
        for i, (sample, id_batch) in enumerate(zip(loader, image_ids)):
            if len(sample) == 0:
                continue
            _, targets = sample
            for target, image_id in zip(targets, id_batch):
                cocosaver.add(target, image_id)
        cocosaver.save(mode)

    @torch.no_grad()
    def infer(self, mode):
        """Run inference on the model.

        Args:
            mode (str): the sample to be evaluated (train/val/infer).
        """
        print('Running inference with mode {}...'.format(mode))
        if mode == 'train':
            loader, image_ids = self.train_loader, self.train_ids
        elif mode in ['val', 'infer']:
            loader, image_ids = self.val_loader, self.val_ids
        else:
            raise NotImplementedError
        cocosaver = COCOSaver(gt=False, cfg=self.cfg)
        self.model.eval()
        for sample, id_batch in tqdm(zip(loader, image_ids)):
            if len(sample) == 0:
                continue
            images, targets = sample
            images_copy = copy.deepcopy(images)
            images = [im.to(self.device) for im in images]
            preds = self.model(images)
            for image, target, pred, image_id in zip(
                    images_copy, targets, preds, id_batch):
                pred['masks'] = (pred['masks'].squeeze(1) >
                                 self.cfg.mask_threshold)
                cocosaver.add(pred, image_id)
                file_name = image_id if mode == 'infer' else None
                self.saver.log_tb_visualization(
                    mode=mode,
                    epoch=self.epoch,
                    image=image,
                    target=target,
                    pred=pred,
                    file_name=file_name)
        cocosaver.save(mode)

    def evaluate(self, mode):
        """Evaluates the saved predicted annotations versus ground truth.

        Args:
            mode (str): the sample to be evaluated (train/val).
        """
        metrics = evaluate(self.cfg, mode)
        self.saver.log_tb_eval(
            mode=mode, metrics=metrics, epoch=self.epoch)
        if mode == 'val':
            self.metrics = metrics
            # flag epoch if it is the best so far
            if self.best_metrics is None:
                self.epoch_is_best = True
            elif (self.metrics[self.cfg.key_metric_name] >
                  self.best_metrics[self.cfg.key_metric_name]):
                self.epoch_is_best = True
            else:
                self.epoch_is_best = False
            # record best
            if self.epoch_is_best:
                self.best_metrics = self.metrics

    def save_checkpoint(self):
        """Saves the checkpoint."""
        self.saver.save_checkpoint(
            state_dict={'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()},
            metrics=self.metrics,
            is_best=self.epoch_is_best)

    def close(self):
        """Properly finish training."""
        self.saver.close_tb_summary()
        print('Finished!')
        print('=' * 72)


if __name__ == '__main__':

    # collect command line arguments
    parser = argparse.ArgumentParser(description='Run Mask RCNN.')
    parser.add_argument('--comment', type=str, default='tmp',
                        help='Name of the run.')
    parser.add_argument('--config', nargs='+', type=str, default=None,
                        help='Specify config files.')
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
    cfg.update([os.path.join('config', f + '.yaml')
                for f in args.config])

    # sanity checks
    assert torch.__version__ >= '1.1.0'
    # update config
    if not args.no_cuda:
        assert torch.cuda.is_available(), 'CUDA not available.'
    cfg.num_gpus = torch.cuda.device_count()
    assert cfg.num_gpus <= args.cuda_max_devices, (
        '{} GPUs available, please set visible devices.\n'
        'export CUDA_VISIBLE_DEVICES=X'
        .format(cfg.num_gpus))
    cfg.batch_size = (cfg.batch_size_per_gpu if cfg.num_gpus == 0
                      else cfg.batch_size_per_gpu * cfg.num_gpus)
    assert os.path.exists(cfg.runs_dir), 'Model/log directory does not exist.'
    if args.resume_run is None:
        cfg.resume_dir = None
    else:
        cfg.resume_dir = os.path.join(cfg.runs_dir, args.resume_run)
        assert os.path.exists(cfg.resume_dir)
    assert not (('infer' in args.mode) and
                (('train' in args.mode) or ('val' in args.mode)))
    cfg.mode = args.mode
    cfg.comment = args.comment
    # construct int-str mapping
    if cfg.int_dict is None:
        cfg.int_dict = {i: name for name, i in cfg.label_dict.items()}

    # train/val/infer starts
    trainer = Trainer(cfg)
    if 'infer' in cfg.mode:
        trainer.infer('infer')
    else:
        for mode in cfg.mode:
            if (mode == 'train') and (not cfg.eval_train):
                continue
            trainer.save_gt_annotations(mode)
            trainer.infer(mode)
            trainer.evaluate(mode)
        if 'train' in cfg.mode:
            # training
            while trainer.epoch < cfg.epochs:
                trainer.train()
                for mode in cfg.mode:
                    if (mode == 'train') and (not cfg.eval_train):
                        continue
                    trainer.infer(mode)
                    trainer.evaluate(mode)
                trainer.save_checkpoint()
    trainer.close()
