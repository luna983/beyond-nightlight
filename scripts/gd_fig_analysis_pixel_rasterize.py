import os
import numpy as np
import pandas as pd

from maskrcnn.postprocess.analysis import load_gd_census, snap_to_grid


np.random.seed(0)

IN_CENSUS_GPS_DIR = 'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv'
IN_CENSUS_MASTER_DIR = 'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta'
IN_SAT_DIR = 'data/Siaya/Merged/sat.csv'

OUT_DIR = 'output/fig-ate'

# define the grid
grid = {
    'min_lon': 34.03,  # 34.067830
    'max_lon': 34.46,  # 34.450290
    'min_lat': -0.06,  # -0.048042
    'max_lat': 0.32,  # 0.317786
    'step': 0.001,  # degrees
}

# MAIN

# load satellite predictions
print('Loading satellite data')
df_sat = pd.read_csv(IN_SAT_DIR)
# create new var
df_sat.loc[:, 'RGB_mean'] = df_sat.loc[:, ['R_mean', 'G_mean', 'B_mean']].mean(axis=1)
# snap to grid
_, df_sat = snap_to_grid(
    df_sat, lon_col='centroid_lon', lat_col='centroid_lat', **grid,
    house_count=pd.NamedAgg(column='area', aggfunc='count'),
    area_sum=pd.NamedAgg(column='area', aggfunc='sum'),
    RGB_mean=pd.NamedAgg(column='RGB_mean', aggfunc='mean'),
)
df_sat.fillna({'house_count': 0, 'area_sum': 0}, inplace=True)
df_sat.loc[:, 'house_count_0'] = (df_sat['house_count'] == 0).values.astype(np.float)

# recover lon, lat
df_sat.loc[:, 'lon'] = df_sat['grid_lon'] * grid['step'] + grid['min_lon'] + grid['step'] / 2
df_sat.loc[:, 'lat'] = df_sat['grid_lat'] * grid['step'] + grid['min_lat'] + grid['step'] / 2

# convert unit
df_sat.loc[:, 'area_sum'] *= ((0.001716 * 111000 / 800) ** 2)  # in sq meters
df_sat.loc[:, 'area_sum_pct'] = df_sat['area_sum'].values / ((grid['step'] * 111000) ** 2)

# load the data frame for treatment status
print('Loading treatment data')
df_treat_raw = load_gd_census(
    GPS_FILE=IN_CENSUS_GPS_DIR, MASTER_FILE=IN_CENSUS_MASTER_DIR)
# snap to grid
_, df_treat = snap_to_grid(
    df_treat_raw, lon_col='longitude', lat_col='latitude', **grid,
    n_household=pd.NamedAgg(column='treat_eligible', aggfunc='count'),
    eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
    treat_eligible=pd.NamedAgg(column='treat_eligible', aggfunc='sum'),
)
df_treat.fillna(0, inplace=True)

# merge treatment with satellite predictions
df = pd.merge(df_treat, df_sat, how='outer', on=['grid_lon', 'grid_lat'])
# drop grids with 0 eligibles
df = df.loc[df['eligible'] > 0, :]

# check treat_eligible distribution
print('treat_eligible: \n', df['treat_eligible'].value_counts())

# save to output
df.to_csv(os.path.join(OUT_DIR, 'main.csv'))

# PLACEBO TEST
for i_simu in range(200):
    # draw saturation level
    df_draw = pd.merge(
        pd.DataFrame({
            'hi_sat': np.random.random(68),
            'satlevel_name': df_treat_raw['satlevel_name'].unique()}),
        df_treat_raw.drop(columns=['hi_sat', 'treat', 'treat_eligible']),
        how='right',
        on='satlevel_name')
    df_draw.loc[:, 'hi_sat'] = (df_draw.loc[:, 'hi_sat'] > 0.5).astype(float)
    # draw treatment status
    df_draw.loc[:, 'treat'] = np.random.random(df_draw.shape[0])
    df_draw.loc[:, 'treat'] = df_draw.apply(
        lambda x: float(x['treat'] > (0.33 if x['hi_sat'] else 0.67)),
        axis=1)
    # create treat x eligible
    df_draw.loc[:, 'treat_eligible'] = df_draw['treat'].values * df_draw['eligible'].values

    # collapse to grid
    _, df_draw = snap_to_grid(
        df_draw, lon_col='longitude', lat_col='latitude', **grid,
        n_household=pd.NamedAgg(column='treat_eligible', aggfunc='count'),
        eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
        treat_eligible=pd.NamedAgg(column='treat_eligible', aggfunc='sum'),
    )
    df_treat.fillna(0, inplace=True)

    # merge treatment with satellite predictions
    df = pd.merge(df_draw, df_sat, how='outer', on=['grid_lon', 'grid_lat'])
    # drop grids with 0 eligibles
    df = df.loc[df['eligible'] > 0, :]
    # save to output
    df.to_csv(os.path.join(OUT_DIR, f'placebo_{i_simu:03d}.csv'))
