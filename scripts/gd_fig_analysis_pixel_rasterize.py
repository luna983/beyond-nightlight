import os
import numpy as np
import pandas as pd

from maskrcnn.postprocess.analysis import (
    load_gd_census, snap_to_grid,
    load_building, load_nightlight_asis)


np.random.seed(0)


def merge_treatment(grid, df_sat, GPS_FILE, MASTER_FILE, OUT_DIR, N=200):
    # load the data frame for treatment status
    print('Loading treatment data')
    df_treat_raw = load_gd_census(
        GPS_FILE=GPS_FILE, MASTER_FILE=MASTER_FILE)
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
    df.to_csv(os.path.join(OUT_DIR, 'main.csv'), index=False)

    # PLACEBO TEST
    for i_simu in range(N):
        # draw saturation level
        df_draw = pd.merge(
            pd.DataFrame({
                'hi_sat': np.random.random(68),
                'satlevel_name': df_treat_raw['satlevel_name'].unique()}),
            df_treat_raw.drop(columns=['hi_sat', 'treat', 'treat_eligible']),
            how='right',
            on='satlevel_name')
        df_draw.loc[:, 'hi_sat'] = (
            df_draw.loc[:, 'hi_sat'] > 0.5).astype(float)
        # draw treatment status
        df_draw.loc[:, 'treat'] = np.random.random(df_draw.shape[0])
        df_draw.loc[:, 'treat'] = df_draw.apply(
            lambda x: float(x['treat'] > (0.33 if x['hi_sat'] else 0.67)),
            axis=1)
        # create treat x eligible
        df_draw.loc[:, 'treat_eligible'] = (
            df_draw['treat'].values * df_draw['eligible'].values)

        # collapse to grid
        _, df_draw = snap_to_grid(
            df_draw, lon_col='longitude', lat_col='latitude', **grid,
            n_household=pd.NamedAgg(column='treat_eligible', aggfunc='count'),
            eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
            treat_eligible=pd.NamedAgg(column='treat_eligible', aggfunc='sum'),
        )
        df_treat.fillna(0, inplace=True)

        # merge treatment with satellite measures
        df = pd.merge(
            df_draw, df_sat, how='outer', on=['grid_lon', 'grid_lat'])
        # drop grids with 0 eligibles
        df = df.loc[df['eligible'] > 0, :]
        # save to output
        df.to_csv(os.path.join(OUT_DIR, f'placebo_{i_simu:03d}.csv'),
                  index=False)


if __name__ == '__main__':

    IN_CEN_GPS_DIR = (
        'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv')
    IN_CEN_MASTER_DIR = (
        'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
    IN_SAT_BD_DIR = 'data/Siaya/Merged/sat.csv'
    IN_SAT_NL_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'

    OUT_DIR_ROOT = 'output/fig-ate/data'

    # main
    grid = {
        'min_lon': 34.03,  # 34.067830
        'max_lon': 34.46,  # 34.450290
        'min_lat': -0.06,  # -0.048042
        'max_lat': 0.32,  # 0.317786
        'step': 0.001,  # degrees
    }
    _, df_sat = load_building(IN_SAT_BD_DIR, grid)
    merge_treatment(
        grid=grid, df_sat=df_sat,
        GPS_FILE=IN_CEN_GPS_DIR, MASTER_FILE=IN_CEN_MASTER_DIR,
        OUT_DIR=os.path.join(OUT_DIR_ROOT, 'building'))

    # nightlight
    grid, df_sat = load_nightlight_asis(IN_SAT_NL_DIR)
    merge_treatment(
        grid=grid, df_sat=df_sat,
        GPS_FILE=IN_CEN_GPS_DIR, MASTER_FILE=IN_CEN_MASTER_DIR,
        OUT_DIR=os.path.join(OUT_DIR_ROOT, 'nightlight'))
