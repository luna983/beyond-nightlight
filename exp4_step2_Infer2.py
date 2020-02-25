from argparse import Namespace

from maskrcnn.train import run

args = Namespace()
args.comment = 'GDInfer'
args.config = ['main', 'siaya2']
args.mode = ['infer']
args.resume_run = 'run_15_Pool'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
