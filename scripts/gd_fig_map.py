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


# SET-UP

IN_BOUND_DIR = 'data/External/GiveDirectly/figure2/SampleArea.shp'
IN_DF_DIR = 'data/Siaya/Merged/main_res0.0050.csv'
OUT_DIR = 'output/fig-map'

# palette for treatment intensity
palette = ['gainsboro', 'yellowgreen', 'darkgreen']
cmap_break = np.linspace(0, 1, len(palette))
cmap_treat = LinearSegmentedColormap.from_list(
    '', list(zip(cmap_break, palette)), N=5)
cmap_treat.set_bad('#ffffff')

# palette for outcome
palette = ['#820000', '#ea0000', '#fff4da', '#5d92c4', '#070490']
cmap_break = np.linspace(0, 1, len(palette))
cmap_outcome = LinearSegmentedColormap.from_list(
    '', list(zip(cmap_break, palette[::-1])), N=10)
cmap_outcome.set_bad('#ffffff')

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

for varname, vmin, vmax, cmap in zip(
    ['treat_eligible', 'area_sum', 'tin_area_sum', 'nightlight'],
    [-0.1, 0, 0, 0],  # vmin
    [11.9, 1.3e4, 0.7e4, 1],  # vmax
    [cmap_treat, cmap_outcome, cmap_outcome, cmap_outcome],
):
    # drop observations with no eligible households
    df.loc[:, varname] = df.apply(
        lambda x: np.nan if x['eligible'] == 0 else x[varname],
        axis=1)
    # convert to raster
    raster = df2raster(df=df, data=varname,
                       row='grid_lat', col='grid_lon')
    # generate plots
    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(raster, origin='lower',
                   extent=(min_lon, max_lon, min_lat, max_lat),
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(*bound.exterior.xy, color='white', linewidth=2)
    ax.add_patch(PolygonPatch(
        outside, facecolor='#dddddd', edgecolor='#dddddd'))
    ax.axis('off')
    ax.set_aspect(1)
    cbar = fig.colorbar(im, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_ticks([])
    fig.savefig(os.path.join(OUT_DIR, varname + '.pdf'),
                bbox_inches='tight', pad_inches=0)
