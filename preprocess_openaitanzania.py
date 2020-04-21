import os
from tqdm import tqdm
from glob import glob
from argparse import Namespace

from maskrcnn.preprocess.preprocess_openaitanzania import process_file


cfg = Namespace()
# define paths
cfg.IN_IMAGE_DIR = 'data/OpenAITanzania/GeoTIFF/'
cfg.IN_ANN_DIR = 'data/OpenAITanzania/GeoJSON/'
cfg.OUT_IMAGE_DIR = 'data/OpenAITanzania/Image/'
cfg.OUT_ANN_DIR = 'data/OpenAITanzania/Mask/'
# parameters
cfg.CHIP_SIZE = 1000  # pixels
cfg.DOWN_RESOLUTION_FACTOR = 4  # resolution = this x 7cm
cfg.WINDOW_SIZE = int(cfg.CHIP_SIZE * cfg.DOWN_RESOLUTION_FACTOR)
cfg.SAMPLE_RATIO = 3
# construct list of training file ids
file_ids = [os.path.basename(f).split('.')[0]
            for f in glob(os.path.join(cfg.IN_ANN_DIR, '*.geojson'))]
# process every file
for file_id in tqdm(file_ids):
    process_file(file_id, cfg)
