from argparse import Namespace

from maskrcnn.train import run

args = Namespace()
args.comment = 'Infer'
args.config = ['main', 'ambo1']
args.mode = 'infer'
args.resume_run = 'run_04_FinetuneGEP'
args.no_cuda = False
args.cuda_max_devices = 1

run(args)
