import argparse
import numpy as np
import pandas as pd
from maskrcnn.postprocess.analysis import (
    winsorize, snap_to_grid,
    load_gd_census, load_building, load_nightlight_from_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution',
                        help='spatial resolution, in degrees',
                        type=float, required=True)
    parser.add_argument('--eligible-only',
                        help='only store grids with eligible households',
                        action='store_true')
    parser.add_argument('--placebo',
                        help='generate placebo runs',
                        type=int, default=0)
    args = parser.parse_args()

    IN_SVY_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
    IN_CENSUS_GPS_DIR = (
        'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv')
    IN_CENSUS_MASTER_DIR = (
        'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
    IN_SAT_BUILDING_DIR = 'data/Siaya/Merged/sat.csv'
    IN_SAT_NIGHTLIGHT_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'

    OUT_DIR = f'data/Siaya/Merged/main_res{args.resolution:.4f}.csv'

    # define the grid
    grid = {
        'min_lon': 34.03,  # 34.067830
        'max_lon': 34.46,  # 34.450290
        'min_lat': -0.06,  # -0.048042
        'max_lat': 0.32,  # 0.317786
        'step': args.resolution,  # degrees
    }

    # load the data frame for treatment status
    df_census = load_gd_census(
        GPS_FILE=IN_CENSUS_GPS_DIR, MASTER_FILE=IN_CENSUS_MASTER_DIR)
    sat_grps = df_census.loc[:, ['satlevel_name']].drop_duplicates()
    villages = df_census.loc[:, ['village_code',
                                 'satlevel_name']].drop_duplicates()
    df_census_placebo = df_census.loc[:, ['longitude', 'latitude',
                                          'eligible',
                                          'village_code', 'satlevel_name']]
    # snap to grid
    _, df_raster_census = snap_to_grid(
        df_census, lon_col='longitude', lat_col='latitude', **grid,
        n_household=pd.NamedAgg(column='treat_eligible', aggfunc='count'),
        eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
        treat_eligible=pd.NamedAgg(column='treat_eligible', aggfunc='sum'),
    )
    df_raster_census.fillna(0, inplace=True)
    df_raster_census = df_raster_census.astype({
        'n_household': int,
        'eligible': int,
        'treat_eligible': int,
    })

    # load satellite predictions
    _, df_raster_building = load_building(IN_SAT_BUILDING_DIR, grid)
    # fill NAs with 0s
    df_raster_building.fillna(0, inplace=True)

    # merge treatment with satellite predictions
    df = pd.merge(df_raster_census, df_raster_building,
                  how='left', on=['grid_lon', 'grid_lat'])
    # load nightlight values
    df = load_nightlight_from_point(
        df, IN_SAT_NIGHTLIGHT_DIR,
        lon_col='lon', lat_col='lat')

    # subset to eligible only
    if args.eligible_only:
        df = df.loc[df['eligible'] > 0, :]
    # winsorize outcome variables
    for varname in ['nightlight', 'area_sum', 'tin_area_sum']:
        df.loc[:, varname] = winsorize(df[varname], 2.5, 97.5)
    # placebo runs
    for i_simu in range(args.placebo):

        # draw saturation level
        df_draw = sat_grps.copy()
        df_draw.loc[:, 'hi_sat'] = (
            np.random.random(sat_grps.shape[0]) > 0.5)
        df_draw = pd.merge(
            villages, df_draw,
            how='left',
            on='satlevel_name')
        # draw treatment status
        df_draw.loc[:, 'treat'] = np.random.random(df_draw.shape[0])
        df_draw.loc[:, 'treat'] = df_draw.apply(
            lambda x: float(x['treat'] > (0.33 if x['hi_sat'] else 0.67)),
            axis=1)
        # merge with full dataset
        df_draw = pd.merge(
            df_census_placebo, df_draw,
            how='left', on=['satlevel_name', 'village_code'])
        # create treat x eligible
        df_draw.loc[:, 'treat_eligible'] = (
            df_draw['treat'].values * df_draw['eligible'].values)

        # collapse to grid
        _, df_draw = snap_to_grid(
            df_draw, lon_col='longitude', lat_col='latitude', **grid,
            n_household=pd.NamedAgg(
                column='treat_eligible', aggfunc='count'),
            eligible=pd.NamedAgg(column='eligible', aggfunc='sum'),
            treat_eligible=pd.NamedAgg(
                column='treat_eligible', aggfunc='sum'),
        )
        df_draw.fillna(0, inplace=True)
        # merge to master data frame
        df_draw = df_draw.loc[:, [
            'grid_lon', 'grid_lat', 'treat_eligible']]
        df_draw = df_draw.rename(
            {'treat_eligible':
                f'treat_eligible_placebo{i_simu:02d}'}, axis=1)
        df_draw = df_draw.astype(int)
        df = pd.merge(df, df_draw, how='left', on=['grid_lon', 'grid_lat'])

    # check treat_eligible distribution
    # print('treat_eligible: \n', df['treat_eligible'].value_counts())
    # print metadata
    print('>>> df.shape')
    print(df.shape)
    print('>>> df.dtypes')
    print(df.dtypes)
    print('-' * 72)
    print(df.describe().T)

    # save to output
    df.to_csv(OUT_DIR, index=False)
