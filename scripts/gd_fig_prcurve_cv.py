import os
import shutil
import glob
import json
from random import shuffle
from argparse import Namespace

from maskrcnn.train import run


k = 3

imgs = list(glob.glob('data/Siaya/Mask/*.json'))
imgs = [os.path.basename(img).split('.')[0] for img in imgs]
shuffle(imgs)
N = len(imgs)

for i in range(k):
    val_imgs = imgs[int(i * N / k):int((i + 1) * N / k)]
    train_imgs = imgs[0:int(i * N / k)] + imgs[int((i + 1) * N / k):]
    
    with open('data/Siaya/train.txt', 'w') as f:
        json.dump(train_imgs, f)
    with open('data/Siaya/val.txt', 'w') as f:
        json.dump(val_imgs, f)

    args = Namespace()
    args.comment = f'SiayaCV{i}'
    args.config = ['main', 'siaya_cv']
    args.mode = ['train', 'val']
    args.resume_run = f'run_{i + 3:02d}_SiayaCV{i}'
    args.no_cuda = False
    args.cuda_max_devices = 1

    run(args)

    shutil.copyfile('data/Siaya/Pred/val/gt.json',
                    f'data/Siaya/Pred/val/gt_cv{i}.json')
    shutil.copyfile('data/Siaya/Pred/val/pred.json',
                    f'data/Siaya/Pred/val/pred_cv{i}.json')

os.remove('data/Siaya/train.txt')
os.remove('data/Siaya/val.txt')
