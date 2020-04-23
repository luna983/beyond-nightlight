from argparse import Namespace

from maskrcnn.train import run


args = Namespace()
args.comment = 'PretrainPool'
args.config = ['main', 'pool']
args.mode = ['train', 'val']
args.resume_run = 'run_01_PretrainPool'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)