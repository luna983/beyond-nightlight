import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf

from .utils import transform_coord


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
        (df.loc[:, by + [column]]
           .groupby(by).transform(np.nanmean).values.squeeze()))


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
    df.loc[:, 'treat_eligible'] = (
        df.loc[:, 'treat'].values * df.loc[:, 'eligible'].values)
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
    assert df['satlevel_name'].notna().all(), (
        'Missing saturation level identifier')
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
    df_grid = pd.DataFrame({'grid_lon': grid_lon.flatten(),
                            'grid_lat': grid_lat.flatten()})

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
    mod = smf.ols(formula=f"z ~ 1 + cr(x, df={cr_df}) + cr(y, df={cr_df})",
                  data=df)
    res = mod.fit()
    # return nan's for cases where any one of x, y, z is nan
    z_out = np.full_like(z, np.nan)
    z_out[~is_na] = z[~is_na] - res.fittedvalues
    return z_out


def load_nightlight_from_point(df, NL_IN_DIR, lon_col='lon', lat_col='lat'):
    # extract nightlight values
    ds = rasterio.open(NL_IN_DIR)
    band = ds.read().squeeze(0)

    idx = np.round(transform_coord(
        transform=ds.transform,
        to='colrow',
        xy=df.loc[:, [lon_col, lat_col]].values)).astype(np.int)

    df.loc[:, 'sat_nightlight'] = [band[i[1], i[0]] for i in idx]
    # winsorize + normalize
    df.loc[:, 'sat_nightlight_winsnorm'] = winsorize(
        df['sat_nightlight'], 0, 99)
    df.loc[:, 'sat_nightlight_winsnorm'] = (
        (df['sat_nightlight_winsnorm'].values -
            np.nanmean(df['sat_nightlight_winsnorm'].values)) /
        np.nanstd(df['sat_nightlight_winsnorm'].values))
    return df


def load_nightlight_asis(input_dir):
    """Loads nightlight data, keeping its raster grid as is.

    Args:
        input_dir (str)
    Returns:
        dict {str: float}: with the following keys
            min_lon, max_lon, min_lat, max_lat, step
        pandas.DataFrame
    """
    # load satellite data
    print('Loading nightlight data')
    ds = rasterio.open(input_dir)
    band = ds.read().squeeze(0)

    # define the grid
    grid = {
        'min_lon': ds.bounds[0],
        'min_lat': ds.bounds[1],
        'max_lon': ds.bounds[2],
        'max_lat': ds.bounds[3],
        'step': ds.transform[0],
    }

    # construct the grid
    grid_lon, grid_lat = np.meshgrid(
        np.arange(0, ds.width),
        np.arange(0, ds.height))

    # convert to data frame
    df = pd.DataFrame({
        'grid_lon': grid_lon.flatten(),
        'grid_lat': grid_lat[::-1].flatten(),
        'nightlight': band.flatten(),
    })

    # recover lon, lat
    df.loc[:, 'lon'] = (
        df['grid_lon'] * grid['step'] + grid['min_lon'] + grid['step'] / 2)
    df.loc[:, 'lat'] = (
        df['grid_lat'] * grid['step'] + grid['min_lat'] + grid['step'] / 2)

    # winsorize + normalize
    df.loc[:, 'nightlight'] = winsorize(
        df['nightlight'], 0, 99)
    df.loc[:, 'nightlight'] = (
        (df['nightlight'].values -
            np.nanmean(df['nightlight'].values)) /
        np.nanstd(df['nightlight'].values))
    return grid, df


def load_building(input_dir, grid, agg=True):
    """Loads building polygons.

    Args:
        input_dir (str): file to load
        grid (dict {str: float}): dict with the following keys:
            min_lon, max_lon, min_lat, max_lat, step
        agg (bool): whether to perform aggregation

    Returns:
        tuple of numpy.ndarray: (grid_lon, grid_lat)
        pandas.DataFrame: gridded dataframe
    """
    tin_roofs = [0, 1, 5]
    thatched_roofs = [2, 3, 6]
    # load satellite predictions
    print('Loading building polygon data')
    df = pd.read_csv(input_dir)
    n_clusters = df['color_group'].max() + 1
    for i in range(n_clusters):
        df.loc[:, f'color_group_{i}'] = (df['color_group'].values == i)
    df.loc[:, 'color_tin'] = df['color_group'].isin(tin_roofs)
    df.loc[:, 'color_thatched'] = df['color_group'].isin(thatched_roofs)
    # tin roof area
    df.loc[:, 'color_tin_area'] = (
        df['color_tin'].values * df['area'].values)
    # thatched roof area
    df.loc[:, 'color_thatched_area'] = (
        df['color_thatched'].values * df['area'].values)
    # create new var: luminosity
    df.loc[:, 'RGB_mean'] = (
        df.loc[:, ['R_mean', 'G_mean', 'B_mean']].mean(axis=1))
    # control for lat lon cubic spline
    df.loc[:, 'RGB_mean_spline'] = control_for_spline(
        x=df['centroid_lon'].values,
        y=df['centroid_lat'].values,
        z=df['RGB_mean'].values,
    )
    # normalize
    df.loc[:, 'RGB_mean_spline'] = (
        (df['RGB_mean_spline'].values -
            np.nanmean(df['RGB_mean_spline'].values)) /
        np.nanstd(df['RGB_mean_spline'].values))
    if not agg:
        return df
    # snap to grid
    color_group_agg = {
        f'color_group_{i}': pd.NamedAgg(
            column=f'color_group_{i}', aggfunc='mean')
        for i in range(n_clusters)}
    (grid_lon, grid_lat), df = snap_to_grid(
        df, lon_col='centroid_lon', lat_col='centroid_lat', **grid,
        house_count=pd.NamedAgg(column='area', aggfunc='count'),
        area_sum=pd.NamedAgg(column='area', aggfunc='sum'),
        RGB_mean=pd.NamedAgg(column='RGB_mean', aggfunc='mean'),
        RGB_mean_spline=pd.NamedAgg(column='RGB_mean_spline', aggfunc='mean'),
        tin_area_sum=pd.NamedAgg(column='color_tin_area', aggfunc='sum'),
        thatched_area_sum=pd.NamedAgg(column='color_thatched_area',
                                      aggfunc='sum'),
        tin_count=pd.NamedAgg(column='color_tin', aggfunc='sum'),
        thatched_count=pd.NamedAgg(column='color_thatched', aggfunc='sum'),
        **color_group_agg,
    )
    df.fillna({'house_count': 0, 'area_sum': 0}, inplace=True)
    df.loc[:, 'house_count_0'] = (
        df['house_count'] == 0).values.astype(np.float)
    df.loc[:, 'area_sum_pct'] = (
        df['area_sum'].values / ((grid['step'] * 111000) ** 2))
    df.loc[:, 'tin_count_pct'] = (
        df['tin_count'].values / df['house_count'].values)
    df.loc[:, 'tin_area_pct'] = (
        df['tin_area_sum'].values / df['area_sum'].values)
    # recover lon, lat
    df.loc[:, 'lon'] = (
        df['grid_lon'] * grid['step'] + grid['min_lon'] + grid['step'] / 2)
    df.loc[:, 'lat'] = (
        df['grid_lat'] * grid['step'] + grid['min_lat'] + grid['step'] / 2)

    return (grid_lon, grid_lat), df
