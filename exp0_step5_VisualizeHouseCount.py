import os
import pandas as pd
from argparse import Namespace

from maskrcnn.postprocess.resample import Resampler

# set up visualization cfg
cfg = Namespace()
cfg.int_dict = {1: 'house'}
cfg.visual_score_cutoff = 0.8
cfg.font = 'maskrcnn/utils/fonts/UbuntuMono-B.ttf'
cfg.bbox_outline = [255, 255, 255, 255]  # white
cfg.bbox_width = 1  # in pixels
cfg.label_fill = [255, 255, 255, 255]  # white
cfg.up_scale = 1
cfg.font_size = 14
cfg.category_palette = {
    1: [255, 140, 105, 64]  # salmon
}
cfg.xmax = 480
cfg.ymax = 770

# AOI index data w/ georeferencing info
AOI_IN_DIR = 'data/Experiment0/aoi.csv'
# download log data
LOG_IN_DIR = 'data/Experiment0/aoi_download_log.csv'
# census data
CEN_IN_DIR = 'data/Experiment0/census.csv'
# satellite images
IMG_IN_DIR = 'data/Mexico/Image/'
# satellite derived annotations
ANN_IN_DIR = 'data/Mexico/Pred/infer/'
# output path
OUT_DIR = 'data/Experiment0/Output/'

# number of visualizations
N_VIZ = 10

# read image index data frame
df = pd.merge(pd.read_csv(AOI_IN_DIR),
              pd.read_csv(LOG_IN_DIR).loc[:, 'index'],
              how='right', on='index')
# read census data
df_cen = pd.read_csv(CEN_IN_DIR, index_col=['ent', 'mun', 'loc'])

# link all inference data
r = Resampler.from_bounds(
    img_dir=IMG_IN_DIR,
    ann_dir=ANN_IN_DIR,
    indices=df['index'].values.tolist(),
    bounds=df[['lon_min', 'lat_min', 'lon_max', 'lat_max']].values.tolist())

# sample localities to visualize
for i in range(N_VIZ):
    sample = df.sample(1, random_state=i).loc[:, ['ent', 'mun', 'loc']]
    df_viz = pd.merge(sample, df, how='left', on=['ent', 'mun', 'loc'])
    bounds = (df_viz['lon_min'].min(), df_viz['lat_min'].min(),
              df_viz['lon_max'].max(), df_viz['lat_max'].max())
    r.plot(bounds=bounds, width=800, height=800, mode='ann', cfg=cfg)
    r.output.save(os.path.join(
        OUT_DIR, 'ENT{:02d}MUN{:03d}LOC{:04d}CEN{:03d}SAT{:03d}.png'
                 .format(int(sample['ent'].values),
                         int(sample['mun'].values),
                         int(sample['loc'].values),
                         int(df_cen.loc[(sample['ent'], sample['mun'], sample['loc']), 'VIVTOT'].values),
                         len(r.annotations))))