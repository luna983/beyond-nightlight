from argparse import Namespace

from maskrcnn.train import run

args = Namespace()
args.comment = 'GDInfer'
args.config = ['main', 'siaya']
args.mode = ['infer']
args.resume_run = 'run_10_GD'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
