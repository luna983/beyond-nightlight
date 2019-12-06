import os
import glob
import json
from random import shuffle
from argparse import Namespace

from maskrcnn.train import run

imgs = glob.glob('data/Mexico/Mask/*.json')
imgs = [os.path.basename(img) for img in imgs]
locs = list(set([img.split('CHIP')[0] for img in imgs]))
shuffle(locs)
print(locs)

for i in range(4):
    train_locs = tuple(locs[0:(2 * i)] + locs[(2 * i + 2):])
    val_locs = tuple(locs[(2 * i):(2 * i + 2)])
    train = [img.split('.')[0] for img in imgs if img.startswith(train_locs)]
    val = [img.split('.')[0] for img in imgs if img.startswith(val_locs)]
    with open('data/Mexico/train.txt', 'w') as f:
        json.dump(train, f)
    with open('data/Mexico/val.txt', 'w') as f:
        json.dump(val, f)

    args = Namespace()
    args.comment = 'CrossEval' + str(i)
    args.config = ['main', 'mexico', 'ft']
    args.mode = ['train', 'val']
    args.resume_run = 'run_0' + str(i + 4) + '_OATMXCV' + str(i)
    args.no_cuda = False
    args.cuda_max_devices = 1

    run(args)
