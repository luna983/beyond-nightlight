from argparse import Namespace
from maskrcnn.preprocess.download_googlestaticmap import run

args = Namespace()
args.log = 'data/Siaya/Meta/aoi_download_log.csv'
args.initialize = 'data/Siaya/Meta/aoi.csv'
args.api_key = 'maskrcnn/preprocess/GOOGLE_API_KEY.txt'
args.num = None
args.download_dir = None

# this run initializes the downloading logs
run(args)

args.initialize = None
args.num = 25000  # daily limit for downloading
args.download_dir = 'data/Siaya/Image'

# this run downloads the images
run(args)
