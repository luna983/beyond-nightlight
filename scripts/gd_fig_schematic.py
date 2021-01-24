import os
import random
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections.abc import Iterable

random.seed(9)
matplotlib.rc('pdf', fonttype=42)


if __name__ == '__main__':

    N = 10  # no. of images sampled
    # image and prediction input directory
    IMG_IN_DIR = 'data/Siaya/Image'
    PRED_IN_DIR = 'data/Siaya/Merged/sat_raw.geojson'
    # AOI index data w/ georeferencing info
    AOI_IN_DIR = 'data/Siaya/Meta/aoi.csv'
    # download log data
    LOG_IN_DIR = 'data/Siaya/Meta/aoi_download_log.csv'
    # boundary
    BOUND_IN_DIR = 'data/External/GiveDirectly/figure2/SampleArea.shp'
    # output
    OUT_DIR = 'output/fig-schematic/'

    print('Reading image metadata, '
          'keeping only images within the GiveDirectly sample area')
    # read boundary shapefile
    bound, = gpd.read_file(BOUND_IN_DIR)['geometry']
    # read image index data frame
    df_meta = pd.merge(pd.read_csv(AOI_IN_DIR),
                       pd.read_csv(LOG_IN_DIR).loc[:, 'index'],
                       how='right', on='index')
    # drop outside geoms
    df_meta = gpd.GeoDataFrame(
        df_meta,
        geometry=gpd.points_from_xy(
            x=(df_meta['lon_min'].values + df_meta['lon_max'].values) / 2,
            y=(df_meta['lat_min'].values + df_meta['lat_max'].values) / 2)
    )
    df_meta = df_meta.loc[df_meta.geometry.within(bound), :]
    # reset index
    df_meta.set_index('index', inplace=True)
    # establish sampling frame
    idxes = df_meta.index.tolist()

    print('Reading predictions')
    # read predictions
    df_sat = gpd.read_file(PRED_IN_DIR)

    print('Sampling')
    # sampling starts
    for i in range(N):
        # sample one image
        idx = random.choice(idxes)
        print(f'sampled: {idx}')
        df_geoms = df_sat.loc[df_sat['index'] == idx, :]
        meta = df_meta.loc[idx, :]

        # copy image
        shutil.copyfile(
            os.path.join(IMG_IN_DIR, f'{idx}.png'),
            os.path.join(OUT_DIR, f'fig-chips-img{i}.png'))

        # plot annotations
        fig, ax = plt.subplots(figsize=(8, 8))
        for _, row in df_geoms.iterrows():
            geoms = (row['geometry']
                     if isinstance(row['geometry'], Iterable)
                     else [row['geometry']])
            for geom in geoms:
                poly = Polygon(
                    np.array(geom.exterior.coords),
                    facecolor=(row['R_mean'] / 255, row['G_mean'] / 255,
                               row['B_mean'] / 255),
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
        fig.savefig(os.path.join(OUT_DIR, f'fig-chips-poly{i}.pdf'),
                    bbox_inches='tight', pad_inches=0, dpi=100)
