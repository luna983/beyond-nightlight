import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import shapely.geometry
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from maskrcnn.postprocess.analysis import df2raster


sns.set(style='ticks', font='Helvetica')


def make_cmap(palette, breaks=None, bad_color='white', N=5):
    if breaks is None:
        breaks = np.linspace(0, 1, len(palette))
    cmap = LinearSegmentedColormap.from_list(
        '', list(zip(breaks, palette)), N=N)
    cmap.set_bad(bad_color)
    return cmap


if __name__ == '__main__':

    # SET-UP
    IN_BOUND_DIR = 'data/External/GiveDirectly/figure2/SampleArea.shp'
    IN_DF_DIR = 'data/Siaya/Merged/main_res0.0050.csv'
    OUT_DIR = 'output/fig-map'
    palettes = {
        'treat_eligible': ['gainsboro', '#0F9D58', '#137333'],
        'area_sum': ['gainsboro', '#DB4437', '#B31412'],
        'tin_area_sum': ['gainsboro', '#4285F4', '#185ABC'],
        'nightlight': ['gainsboro', '#F4B400', '#EA8600'],
    }
    v_extents = {
        'treat_eligible': (0, 12),
        'area_sum': (0, 1.5e4),
        'tin_area_sum': (0, 0.9e4),
        'nightlight': (0, 1.8),
    }

    # load data
    df = pd.read_csv(IN_DF_DIR)
    print(df.describe().T)
    step_lon = (df['lon'].max() - df['lon'].min()) / df['grid_lon'].max()
    step_lat = (df['lon'].max() - df['lon'].min()) / df['grid_lon'].max()
    min_lon = df['lon'].min() - step_lon / 2
    max_lon = df['lon'].max() + step_lon / 2
    min_lat = df['lat'].min() - step_lat / 2
    max_lat = df['lat'].max() + step_lat / 2

    # load sample area / towns geometry
    bound, = gpd.read_file(IN_BOUND_DIR)['geometry']
    outside = shapely.geometry.box(
        min_lon, min_lat, max_lon, max_lat).difference(bound)

    # generate plots
    fig, axes = plt.subplots(figsize=(6.5, 7), nrows=2, ncols=2)
    for i, varname in enumerate(['treat_eligible', 'area_sum',
                                 'tin_area_sum', 'nightlight']):
        ax = axes[i // 2, i % 2]
        vmin, vmax = v_extents[varname]
        cmap = make_cmap(palettes[varname], breaks=[0, 0.7, 1])
        # drop observations with no eligible households
        df.loc[:, varname] = df.apply(
            lambda x: np.nan if x['eligible'] == 0 else x[varname],
            axis=1)
        # convert to raster
        raster = df2raster(df=df, data=varname,
                           row='grid_lat', col='grid_lon')
        im = ax.imshow(raster, origin='lower',
                       extent=(min_lon, max_lon, min_lat, max_lat),
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(*bound.exterior.xy, color='#cccccc', linewidth=2)
        ax.add_patch(PolygonPatch(
            outside, facecolor='white', edgecolor='white'))
        ax.axis('off')
        ax.set_aspect(1)
        cbar = fig.colorbar(
            im, shrink=0.6, aspect=15,
            orientation='horizontal', ax=ax)
        cbar.outline.set_visible(False)
        cbar.set_ticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig-map-raw.pdf'))
