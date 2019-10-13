from argparse import Namespace

from maskrcnn.train import run

args = Namespace()
args.comment = 'GDInfer'
args.config = ['main', 'givedirectly']
args.mode = 'infer'
args.resume_run = 'run_07_GD'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
