from argparse import Namespace

from maskrcnn.train import run


args = Namespace()
args.comment = 'PretrainOAT'
args.config = ['main', 'openaitanzania']
args.mode = ['train', 'val']
args.resume_run = None
args.no_cuda = False
args.cuda_max_devices = 1

run(args)