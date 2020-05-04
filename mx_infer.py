from argparse import Namespace

from maskrcnn.train import run


args = Namespace()
args.comment = 'MXInfer'
args.config = ['main', 'mexico', 'infer']
args.mode = ['infer']
args.resume_run = 'run_01_PretrainPool'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
