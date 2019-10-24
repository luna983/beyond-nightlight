from argparse import Namespace
from maskrcnn.preprocess.download_googlestaticmap import run

args = Namespace()
args.log = 'data/Experiment2/aoi_download_log.csv'
args.initialize = 'data/Experiment2/aoi.csv'
args.api_key = 'maskrcnn/preprocess/GOOGLE_API_KEY.txt'
args.num = None
args.download_dir = None

# this run initializes the downloading logs
run(args)

args.initialize = None
args.num = 20000
args.download_dir = 'data/Ambo/Image'

# this run downloads the images
run(args)
