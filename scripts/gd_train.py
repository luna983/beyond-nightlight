from argparse import Namespace

from maskrcnn.train import run


args = Namespace()
args.comment = 'Siaya'
args.config = ['main', 'siaya']
args.mode = ['train', 'val']
args.resume_run = 'run_02_Siaya'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)