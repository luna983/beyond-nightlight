
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from argparse import Namespace

from sklearn.decomposition import PCA

from maskrcnn.postprocess.validate import L, gini, plot_scatter
from maskrcnn.postprocess.polygonize import load_anns

from scipy.spatial.distance import cdist

import matplotlib
matplotlib.rc('pdf', fonttype=42)


# In[2]:


def n_neighbor(coords, h):
    return (cdist(coords, coords, 'euclidean') < h).sum(axis=0).mean()

def diff(x):
    return np.max(x) - np.min(x)


# In[3]:


# image index data
IDX_IN_DIR = 'data/Experiment3/aoi.csv'
# download log data
LOG_IN_DIR = 'data/Experiment3/aoi_download_log.csv'

# census data
CEN_IN_DIR = 'data/Experiment3/census.shp'

# satellite derived data
SAT_IN_ANN_DIR = 'data/MexicoInequality/Pred/infer/'
SAT_IN_IMG_DIR = 'data/MexicoInequality/Image/'
SAT_IN_SHP_DIR = 'data/Experiment3/sat.shp'

# output path
OUT_DIR = 'data/Experiment3/Output/'


# In[4]:


# read image index data frame
df_idx = pd.merge(pd.read_csv(IDX_IN_DIR), pd.read_csv(LOG_IN_DIR),
                  how='outer', on='index')
df_idx = df_idx.loc[:, ['index', 'ent', 'mun', 'loc', 'chip',
                        'status', 'lon_min', 'lon_max', 'lat_min', 'lat_max']]


# In[5]:


assert (df_idx['status'] == False).sum() == 0  # no missing


# In[6]:


# read census data
df_cen = gpd.read_file(CEN_IN_DIR)


# In[7]:


if os.path.isfile(SAT_IN_SHP_DIR):
    df_sat = gpd.read_file(SAT_IN_SHP_DIR)
else:
    # read sat annotations
    ann_files = glob.glob(SAT_IN_ANN_DIR + '*.json')
    img_files = [os.path.join(SAT_IN_IMG_DIR,
                              (os.path.relpath(f, SAT_IN_ANN_DIR).replace('.json', '.png')))
                 for f in ann_files]
    df_sat = load_anns(ann_files=ann_files,
                       img_files=img_files,
                       idx_file=IDX_IN_DIR)
    df_sat = pd.concat([
        df_sat.reset_index(drop=True),
        pd.DataFrame(df_sat.loc[:, 'RGB_mean'].tolist(),
                     columns=['R_mean', 'G_mean', 'B_mean'])], axis=1)
    df_sat = pd.merge(df_sat, df_idx, how='left', on='index')
    df_sat = df_sat.drop(columns=['RGB_mean'])

    df_sat = pd.merge(
        df_sat,
        (df_cen.loc[:, ['ent', 'mun', 'loc', 'geometry']]
         .rename({'geometry': 'geometry_loc'}, axis=1).drop_duplicates(['ent', 'mun', 'loc'])),
        how='left', on=['ent', 'mun', 'loc'])

    df_sat = df_sat.loc[df_sat['geometry'].within(gpd.GeoSeries(df_sat['geometry_loc'])), :]

    df_sat = df_sat.drop(columns=['geometry_loc'])

    df_sat.to_file(SAT_IN_SHP_DIR, index=False)


# In[ ]:


# grouping into localities
df_group = df_sat.drop(columns=['geometry']).groupby(['ent', 'mun', 'loc']).agg(
    sat_house=pd.NamedAgg(column='area', aggfunc='count'),
    sat_size_mean=pd.NamedAgg(column='area', aggfunc=np.nanmean),
#     sat_size_med=pd.NamedAgg(column='area', aggfunc=np.nanmedian),  # nah not better than mean
    sat_lum_mean=pd.NamedAgg(column='luminosity', aggfunc=np.nanmean),
    sat_saturation_mean=pd.NamedAgg(column='saturation', aggfunc=np.nanmean),
#     sat_size_gini=pd.NamedAgg(column='area', aggfunc=gini_series),  # no signal
    sat_size_sum=pd.NamedAgg(column='area', aggfunc=np.nansum)
)

# measure spatial clustering
tile_size = 1.6516e-3

for i, col in zip([.3, .53],
                  ['sat_nn_h30', 'sat_nn_h53']):
    df_group[col] = df_sat.groupby(['ent', 'mun', 'loc']).apply(
        lambda grp: n_neighbor(
            np.array([grp.centroid.x.values, grp.centroid.y.values]).T,
            h=tile_size * i))

# calculate K function values
# tile_size = 1.6516e-3
# for i, col in zip([.5, 1, 1.5],
#                   ['sat_dist_h05', 'sat_dist_h10', 'sat_dist_h15']):
#     df_group[col] = df_sat.groupby(['ent', 'mun', 'loc']).apply(
#         lambda grp: L(grp[['lon_center', 'lat_center']].values,
#                       A=(tile_size ** 2) * 25, h=tile_size * i))

# scale areas / distances
df_group[['sat_size_mean', 'sat_size_sum']] *= (
    ((tile_size / 800 * 111000) ** 2) * np.cos(23 / 180 * np.pi))  # in sq meters

df_group['sat_angle_align'] = df_sat.groupby(['ent', 'mun', 'loc']).apply(
    lambda grp: np.sort(np.histogram(
        grp['angle'].values,
        bins=9, range=(0, 90), density=True)[0] * 10)[-2:].sum())

# merge satellite and census
df_group = pd.merge(
    df_group,
    df_cen,
    how='right', on=['ent', 'mun', 'loc'])


# In[ ]:


df_group['rich'] = (df_group['diff'] > 0)


# In[ ]:


sat_cols = [col for col in df_group.columns if col.startswith('sat')]
cen_cols = [col for col in df_group.columns if col.startswith('cen')]


# In[ ]:


df_group = df_group.pivot(index='pair_id', columns='rich', values=['diff'] + sat_cols)


# In[ ]:


for var in sat_cols:
    fig, ax = plt.subplots(figsize=(15, 5))
    # for v in (True, False):
    #     ax.scatter(df_group.loc[:, ('diff', True)], df_group.loc[:, (var, v)], color='k')
    for _, row in df_group.iterrows():
        ax.plot([row[('diff', True)]] * 2, [row[(var, True)], row[(var, False)]],
                color='r' if row[(var, True)] > row[(var, False)] else 'b')
    ax.set_ylabel(var)
    plt.savefig(os.path.join(OUT_DIR, '{}.pdf'.format(var)))

