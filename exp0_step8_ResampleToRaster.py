import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace

from maskrcnn.postprocess.resample import Resampler

# AOI index data w/ georeferencing info
AOI_IN_DIR = 'data/Experiment0/aoi.csv'
# download log data
LOG_IN_DIR = 'data/Experiment0/aoi_download_log.csv'
# satellite derived annotations
ANN_IN_DIR = 'data/Mexico/Pred/infer/'
# output path
OUT_DIR = 'data/Experiment0/Output/'

# set up visualization cfg
cfg = Namespace()
cfg.visual_score_cutoff = 0.8
cfg.xmax = 480
cfg.ymax = 770

# read image index data frame
df = pd.merge(pd.read_csv(AOI_IN_DIR),
              pd.read_csv(LOG_IN_DIR).loc[:, 'index'],
              how='right', on='index')

# link all inference data
r = Resampler.from_bounds(
    ann_dir=ANN_IN_DIR,
    indices=df['index'].values.tolist(),
    bounds=df[['lon_min', 'lat_min', 'lon_max', 'lat_max']].values.tolist())

# specify bounds
bounds = [-103, 20, -102, 21]

# aggregate
r.agg(bounds=bounds,
      width=10,
      height=10,
      f_ann=lambda x: x['score'],
      f_agg=len,
      mode='ann',
      cfg=cfg)

# plot
plt.imshow(r.output)
plt.colorbar()
plt.savefig(os.path.join(OUT_DIR, 'raster.pdf'))