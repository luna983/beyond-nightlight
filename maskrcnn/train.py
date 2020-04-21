import os
import json
import copy
from tqdm import tqdm

import torch

from .dataloader import make_data_loader
from .model import make_model
from .utils.configure import Config
from .utils.save_ckpt_log_tb import Saver
from .utils.coco import COCOSaver
from .utils.eval import evaluate


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
            if self.cfg.visual_train:
                images_copy = copy.deepcopy(images)
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
            if self.cfg.visual_train:
                self.saver.log_tb_visualization(
                    mode='train',
                    epoch=self.epoch,
                    image=images_copy[0],
                    target=targets[0],
                    pred=None)
        self.saver.log_tb_loss(mode='train', losses=losses,
                               loss_dicts=loss_dicts, epoch=self.epoch)

    def save_gt_annotations(self, mode='val'):
        """Saves the ground truth annotations as COCO format.

        Args:
            mode (str): the sample to be evaluated (val).
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
                if mode == 'infer':
                    cocosaver.save(mode, file_name=image_id)
                self.saver.log_tb_visualization(
                    mode=mode,
                    epoch=self.epoch,
                    image=image,
                    target=target,
                    pred=pred,
                    file_name=image_id if mode == 'infer' else None)
        if mode != 'infer':
            cocosaver.save(mode)

    def evaluate(self, mode='val'):
        """Evaluates the saved predicted annotations versus ground truth.

        Args:
            mode (str): the sample to be evaluated (val).
        """
        self.metrics = evaluate(self.cfg, mode)
        self.saver.log_tb_eval(
            mode=mode, metrics=self.metrics, epoch=self.epoch)
        # flag epoch if it is the best so far
        if self.best_metrics is None:
            self.epoch_is_best = True
        elif self.metrics is None:
            self.epoch_is_best = False
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


def run(args):
    """Runs the main training script.

    Args:
        args (argparse.Namespace): all training params
    """

    # parse configurations
    cfg = Config()
    cfg.update([os.path.join('maskrcnn/config', f + '.yaml')
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
    # infer num_classes, include background
    int_set = set(cfg.label_dict.values())
    int_set.add(0)
    cfg.num_classes = len(int_set)
    if cfg.fillempty:
        cfg.label_dict['placeholder'] = cfg.num_classes
        cfg.int_dict[cfg.num_classes] = 'placeholder'
        cfg.num_classes += 1

    # train/val/infer starts
    trainer = Trainer(cfg)
    if 'infer' in cfg.mode:
        trainer.infer('infer')
    else:
        if 'val' in cfg.mode:
            trainer.save_gt_annotations()
            trainer.infer('val')
            trainer.evaluate()
        if 'train' in cfg.mode:
            # training
            while trainer.epoch < cfg.epochs:
                trainer.train()
                if 'val' in cfg.mode:
                    trainer.infer('val')
                    trainer.evaluate()
                trainer.save_checkpoint()
    trainer.close()
