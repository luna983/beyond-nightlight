import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def winsorize(s, lower, upper):
    """Winsorizes a pandas series.
    
    Args:
        s (pandas.Series): the series to be winsorized
        lower, upper (int): number between 0 to 100
    """
    lower_value = np.nanpercentile(s.values, lower)
    upper_value = np.nanpercentile(s.values, upper)
    print(f'Winsorizing to {lower_value} - {upper_value}')
    return s.clip(lower_value, upper_value)


def demean(df, column, by):
    """Demean a column in a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): data
        column (str): the column to be demeaned
        by (list of str): the column names
    """
    return (
        df[column].values -
        df.loc[:, by + [column]].groupby(by).transform(np.nanmean).values.squeeze())


def load_gd_census(GPS_FILE, MASTER_FILE):
    # read GPS coords + treatment status
    df = pd.read_csv(
        GPS_FILE,
        usecols=['village_code', 'ge', 'hi_sat', 'treat',
                 'latitude', 'longitude', 'elevation', 'accuracy', 'eligible',
                 'GPS_imputed'],
        dtype={
            'village_code': 'Int64',
            'ge': 'Int32',
            'hi_sat': 'Int32',
            'treat': 'Int32',
            'eligible': 'Int32',
            'GPS_imputed': 'Int32'})
    # drop non GE households
    df = df.loc[df['ge'] == 1, :].copy()
    # treat x eligible = cash inflow
    df.loc[:, 'treat_eligible'] = df.loc[:, 'treat'].values * df.loc[:, 'eligible'].values
    # read sat level identifiers
    df_master = pd.read_stata(
        MASTER_FILE,
        columns=['village_code', 'satlevel_name']
    ).astype({'village_code': 'Int64'})
    df_master = df_master.drop_duplicates()
    # merge treatment
    df = pd.merge(
        df, df_master,
        on='village_code', how='left')
    assert df['satlevel_name'].notna().all(), 'Missing saturation level identifier'
    return df.drop(columns=['ge'])


def snap_to_grid(df, lon_col, lat_col,
                 min_lon, max_lon, min_lat, max_lat, step,
                 **kwargs):
    """Collapses variables in a data frame onto a grid.

    Args:
        df (pandas.DataFrame)
        lon_col, lat_col (str): name of lon, lat columns
        min_lon, max_lon, min_lat, max_lat, step (float)
        **kwargs: passed to pandas agg() function after grouping by lat, lon

    Returns:
        (numpy.ndarray, numpy.ndarray): lon and lat grids
        pandas.DataFrame: output data frame
    """
    df_copy = df.copy()

    # snap to grid
    df_copy.loc[:, 'grid_lon'] = np.round(
        (df[lon_col].values - min_lon - step / 2) / step
    ).astype(np.int32)
    df_copy.loc[:, 'grid_lat'] = np.round(
        (df[lat_col].values - min_lat - step / 2) / step
    ).astype(np.int32)

    # construct the grid
    grid_lon, grid_lat = np.meshgrid(
        np.arange(0, np.round((max_lon - min_lon) / step).astype(np.int32)),
        np.arange(0, np.round((max_lat - min_lat) / step).astype(np.int32)))
    df_grid = pd.DataFrame({'grid_lon': grid_lon.flatten(), 'grid_lat': grid_lat.flatten()})

    # collapse
    df_output = pd.merge(
        df_grid.assign(is_in_grid=True),
        df_copy.groupby(['grid_lon', 'grid_lat']).agg(**kwargs),
        how='outer', on=['grid_lon', 'grid_lat'])
    print(f"Dropping {df_output['is_in_grid'].isna().sum()} observations;\n"
          f"Keeping {df_output['is_in_grid'].notna().sum()} observations")
    df_output = df_output.loc[df_output['is_in_grid'].notna(), :].copy()

    return (grid_lon, grid_lat), df_output.drop(columns=['is_in_grid'])


def control_for_spline(x, y, z, cr_df=3):
    # handle nan's
    is_na = np.any((np.isnan(x), np.isnan(y), np.isnan(z)), axis=0)
    df = pd.DataFrame({'x': x[~is_na], 'y': y[~is_na], 'z': z[~is_na]})
    mod = smf.ols(formula=f"z ~ 1 + cr(x, df={cr_df}) + cr(y, df={cr_df})", data=df)
    res = mod.fit()
    # return nan's for cases where any one of x, y, z is nan
    z_out = np.full_like(z, np.nan)
    z_out[~is_na] = z[~is_na] - res.fittedvalues
    return z_out
