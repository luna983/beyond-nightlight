import os
import random
import glob
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import shapely
import shapely.geometry
from descartes import PolygonPatch
from collections.abc import Iterable


matplotlib.rc('pdf', fonttype=42)

N = 10  # no. of images sampled

IN_IMG_DIR = 'data/Siaya/Image'
IN_META_DIR = 'data/Siaya/Meta/aoi.csv'
IN_PRED_DIR = 'data/Siaya/Merged/sat.geojson'
OUT_DIR = 'output/fig-schematic/'

# read predictions
df_sat = gpd.read_file(IN_PRED_DIR)
# read meta data
df_meta = pd.read_csv(IN_META_DIR)
df_meta = df_meta.set_index('index')
# establish sampling frame
idxes = [os.path.basename(idx).split('.')[0]
         for idx in glob.glob(os.path.join(IN_IMG_DIR, '*.png'))]

# sampling starts
for _ in range(N):
    # sample one image
    idx = random.choice(idxes)
    df_geoms = df_sat.loc[df_sat['index'] == idx, :]
    meta = df_meta.loc[idx, :]

    # copy image
    shutil.copyfile(
        os.path.join(IN_IMG_DIR, f'{idx}.png'),
        os.path.join(OUT_DIR, f'{idx}_img.png'))

    # plot annotations
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, row in df_geoms.iterrows():
        geoms = (row['geometry']
                 if isinstance(row['geometry'], Iterable)
                 else [row['geometry']])
        for geom in geoms:
            poly = Polygon(
                np.array(geom.exterior.coords),
                facecolor=(row['R_mean'] / 255, row['G_mean'] / 255, row['B_mean'] / 255),
                edgecolor='white',
                linewidth=2,
            )
            ax.add_patch(poly)
    ax.set_facecolor('#DDDDDD')
    ax.set_xlim(meta['lon_min'], meta['lon_max'])
    ax.set_ylim(meta['lat_min'], meta['lat_max'])
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, f'{idx}_color.pdf'))
