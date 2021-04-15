import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import shapely.geometry
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib

from maskrcnn.postprocess.analysis import df2raster


sns.set(style='ticks', font='Helvetica')
matplotlib.rc('pdf', fonttype=42)

plt.rc('font', size=11)  # controls default text sizes
plt.rc('axes', titlesize=11)  # fontsize of the axes title
plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
plt.rc('legend', fontsize=11)  # legend fontsize
plt.rc('figure', titlesize=11)  # fontsize of the figure title


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
    OUT_FIG_DIR = 'output/fig-map/fig-map-raw.pdf'
    OUT_DATA_DIR = 'fig_raw_data/fig-map.csv'

    # load data
    df = pd.read_csv(IN_DF_DIR)
    # print(df.describe().T)
    print('Treatment Households: {}; Control Households: {}'.format(
        df['treat_eligible'].sum(),
        df['eligible'].sum() - df['treat_eligible'].sum()
    ))
    step_lon = (df['lon'].max() - df['lon'].min()) / df['grid_lon'].max()
    step_lat = (df['lon'].max() - df['lon'].min()) / df['grid_lon'].max()
    grid_size_in_m2 = (step_lat * 111000) * (step_lon * 111000)

    min_lon = df['lon'].min() - step_lon / 2
    max_lon = df['lon'].max() + step_lon / 2
    min_lat = df['lat'].min() - step_lat / 2
    max_lat = df['lat'].max() + step_lat / 2

    # load sample area / towns geometry
    bound, = gpd.read_file(IN_BOUND_DIR)['geometry']
    outside = shapely.geometry.box(
        min_lon, min_lat, max_lon, max_lat).difference(bound)

    # styling
    varnames = ['treat_eligible', 'area_sum',
                'tin_area_sum', 'nightlight']
    palettes = {
        'treat_eligible': ['gainsboro', '#0F9D58', '#137333'],
        'area_sum': ['gainsboro', '#DB4437', '#B31412'],
        'tin_area_sum': ['gainsboro', '#4285F4', '#185ABC'],
        'nightlight': ['gainsboro', '#F4B400', '#EA8600'],
    }
    num_colors = 5
    panel_titles = {
        'treat_eligible': 'a    Treatment Intensity',
        'area_sum': 'b    Building Footprint',
        'tin_area_sum': 'c    Tin-roof Area',
        'nightlight': 'd    Night Light',
    }
    output_varname = {
        'treat_eligible': 'treatment_intensity',
        'area_sum': 'building_footprint',
        'tin_area_sum': 'tin_roof_area',
        'nightlight': 'night_light',
    }
    v_extents = {
        'treat_eligible': (0, 15 + 1e-4),
        'area_sum': (0, 5 / 100 * grid_size_in_m2),
        'tin_area_sum': (0, 5 / 100 * grid_size_in_m2),
        'nightlight': (0, 2),
    }
    v_tickformats = {
        'treat_eligible': lambda x: int(x),
        'area_sum': lambda x: f'{x / grid_size_in_m2 * 100:.0f}%',
        'tin_area_sum': lambda x: f'{x / grid_size_in_m2 * 100:.0f}%',
        'nightlight': lambda x: f'{x:.1f}',
    }
    # raw data output
    raw_data_output = df.loc[df['eligible'] > 0,
                             ['lon', 'lat'] + varnames].copy()
    # generate plots
    fig, axes = plt.subplots(figsize=(6.5, 6), nrows=2, ncols=2)
    for i, varname in enumerate(varnames):
        ax = axes[i // 2, i % 2]
        vmin, vmax = v_extents[varname]
        v_tick = np.linspace(vmin, vmax, num_colors + 1)[0:-1]
        v_tickformat = v_tickformats[varname]
        v_ticklabel = [v_tickformat(label) for label in v_tick]
        # output raw data
        v_tick_with_end = [np.NINF] + v_tick[1:].tolist() + [np.PINF]
        v_ticklabel_with_end = v_ticklabel + ['']
        if varname == 'treat_eligible':
            v_rangelabel = ['{}-{}'.format(
                v_ticklabel_with_end[i] + (1 if i > 0 else 0),
                v_ticklabel_with_end[i + 1],
            ) for i in range(num_colors)]
        else:
            v_rangelabel = ['{}-{}'.format(
                v_ticklabel_with_end[i],
                v_ticklabel_with_end[i + 1],
            ) for i in range(num_colors)]
        # left exclusive, right inclusive when binning
        raw_data_output.loc[:, varname] = pd.cut(
            raw_data_output[varname],
            bins=v_tick_with_end, labels=v_rangelabel)
        # make plot
        cmap = make_cmap(palettes[varname], breaks=[0, 0.7, 1], N=num_colors)
        # drop observations with no eligible households
        df.loc[df['eligible'] == 0, varname] = np.nan
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
        ax.set_title(panel_titles[varname], loc='left')
        cbar = fig.colorbar(
            im, location='left', fraction=0.06, aspect=8,
            ax=[ax], anchor=(0, 1), panchor=(0, 1))
        cbar.outline.set_visible(False)
        cbar.set_ticks(v_tick)
        cbar.ax.set_yticklabels(v_ticklabel, ha='left')
        cbar.ax.tick_params(size=0, pad=-15)
    # data output
    raw_data_output = raw_data_output.rename(output_varname, axis=1)
    raw_data_output.loc[:, 'lon'] = raw_data_output['lon'].round(4)
    raw_data_output.loc[:, 'lat'] = raw_data_output['lat'].round(4)
    raw_data_output.to_csv(OUT_DATA_DIR, index=False)
    # plot output
    plt.subplots_adjust(
        left=0, right=1, bottom=0, top=0.9,
        wspace=0, hspace=0.2)
    fig.savefig(OUT_FIG_DIR)
