from argparse import Namespace

from maskrcnn.train import run

args = Namespace()
args.comment = 'GD'
args.config = ['main', 'givedirectly', 'ft']
args.mode = ['train', 'val']
args.resume_run = 'run_10_GD'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
