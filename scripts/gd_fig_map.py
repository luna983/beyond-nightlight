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

from maskrcnn.postprocess.analysis import (
    load_gd_census, snap_to_grid, load_building)


sns.set(style='ticks', font='Helvetica', font_scale=1)


def plot(raster, file,
         cmap, vmin, vmax,
         bound,
         min_lon, max_lon, min_lat, max_lat, step,
         add_polygons=None):
    outside = shapely.geometry.box(
        min_lon, min_lat, max_lon, max_lat).difference(bound)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(raster,
                   extent=(min_lon, max_lon, min_lat, max_lat),
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(*bound.exterior.xy, color='white', linewidth=5)
    ax.add_patch(PolygonPatch(
        outside, facecolor='#dddddd', edgecolor='#dddddd'))
    if add_polygons is not None:
        for polygon, color in add_polygons:
            ax.add_patch(PolygonPatch(polygon, edgecolor=color))
    ax.axis('off')
    ax.set_aspect(1)
    fig.colorbar(im)
    fig.savefig(file)


# SET-UP

IN_BOUND_DIR = 'data/External/GiveDirectly/figure2/SampleArea.shp'
IN_CENSUS_GPS_DIR = ('data/External/GiveDirectly/'
                     'GE_HH_Census_2017-07-17_cleanGPS.csv')
IN_CENSUS_MASTER_DIR = (
    'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
IN_SAT_DIR = 'data/Siaya/Merged/sat.csv'

OUT_DIR = 'output/fig-map'

# define the grid
grid = {
    'min_lon': 34.03,  # 34.067830
    'max_lon': 34.46,  # 34.450290
    'min_lat': -0.06,  # -0.048042
    'max_lat': 0.32,  # 0.317786
    'step': 0.005,  # degrees
}

palette = ['#820000', '#ea0000', '#fff4da', '#5d92c4', '#070490']
cmap_break = np.linspace(0, 1, len(palette))
cmap = LinearSegmentedColormap.from_list(
    '', list(zip(cmap_break, palette[::-1])), N=10)
cmap.set_bad('#ffffff')

# load sample area / towns geometry
bound, = gpd.read_file(IN_BOUND_DIR)['geometry']

# PLOT TREATMENT
# load the data frame for treatment status
df_treat = load_gd_census(
    GPS_FILE=IN_CENSUS_GPS_DIR, MASTER_FILE=IN_CENSUS_MASTER_DIR)
# snap to grid
(grid_lon, grid_lat), df_raster = snap_to_grid(
    df_treat, lon_col='longitude', lat_col='latitude', **grid,
    n_household=pd.NamedAgg(column='treat_eligible', aggfunc='count'),
    eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
    treat_eligible=pd.NamedAgg(column='treat_eligible', aggfunc='sum'),
)
# df_raster.fillna(0, inplace=True)
# # construct pct of eligible households treated
# df_raster.loc[:, 'treat_pct'] = df_raster.apply(
#     lambda x: np.nan if x['eligible'] == 0 else (
#         x['treat_eligible'] / x['eligible']),
#     axis=1)

# plotting begins

# convert to raster
raster = (df_raster['treat_eligible'].astype('float').values
                                     .reshape(grid_lon.shape)[::-1, :])
plot(
    raster=raster,
    file=os.path.join(OUT_DIR, 'treat_eligible.pdf'),
    cmap=cmap, vmin=-9.9, vmax=10,
    bound=bound, **grid)

# PLOT OUTCOMES

# load satellite predictions
(grid_lon, grid_lat), df_raster = load_building(IN_SAT_DIR, grid)

# plotting begins
for outcome, vmin, vmax in zip(
    ['area_sum_pct', 'tin_area_sum_pct'],  # outcome
    [0, 0],  # vmin
    [0.04, 0.025],  # vmax
):
    raster = df_raster[outcome].values.reshape(grid_lon.shape)[::-1, :]
    plot(
        raster=raster,
        file=os.path.join(OUT_DIR, outcome + '.pdf'),
        cmap=cmap, vmin=vmin, vmax=vmax,
        bound=bound, **grid)
