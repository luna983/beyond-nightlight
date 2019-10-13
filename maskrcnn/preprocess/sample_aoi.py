import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def tile_width_in_degree(latitude, zoom_level=19,
                         width=640, height=640):
    """Calculates the width/height of Google Static Map images.

    This is based on the Pseudo Mercator Projection.

    Args:
        latitude (float): the latitude where the image is in.
        zoom_level (int): the zoom level of the image.
        width, height (int): the width, height of the image, in pixels.

    Returns:
        lon_degree, lat_degree (float): the width, height of the image,
            in degrees.
    """

    phi = latitude * np.pi / 180  # in radians
    lon_degree = 360 / 256 / (2 ** zoom_level) * width  # in degrees
    lat_degree = np.abs(360 / 256 / (2 ** zoom_level) * height *
                        2 * np.sin(np.pi / 4 + phi / 2) *
                        np.cos(np.pi / 4 + phi / 2))  # in degrees
    return lon_degree, lat_degree


def dms_to_dd(n):
    """Convert degree minute second to decimal degrees."""
    d = np.floor(n / 10000)
    m = np.floor((n - d * 10000) / 100)
    s = np.floor(n - d * 10000 - m * 100)
    dd = d + m / 60 + s / 3600
    return dd


def load_df(in_dir, drop=True,
            urban_cutoff=0.1, rural_cutoff=0.01,
            dataset='ITER2010'):
    """Loads the ITER 2010 data frame.

    Args:
        in_dir (str): data directory
        drop (bool): whether to subsample
        urban_cutoff, rural_cutoff (float): cutoff values for
            dropping observations too close to each other
            in degrees
        dataset (str)

    Returns:
        pandas.DataFrame: loaded data frame
    """
    assert dataset in ['ITER2010'], 'Unknown dataset specified'
    # specify data type for faster I/O
    df = pd.read_csv(in_dir, nrows=0)

    # select key variables and specify dtypes
    main_cols = ['ENTIDAD', 'MUN', 'LOC',
                 'LONGITUD', 'LATITUD',
                 'POBTOT', 'VIVTOT', 'TVIVHAB', 'TAM_LOC']
    vph_cols = [col for col in df.columns if col.startswith('VPH')]

    # read data frame
    # parse * (masked for privacy) or N/D (information not collected) as NA
    df = pd.read_csv(in_dir, usecols=main_cols + vph_cols,
                     na_values=['*', 'N/D'], dtype='Int32')

    # drop obs without coordinates (regional aggregates)
    # drop obs without census measures
    df.dropna(subset=['LONGITUD', 'LATITUD', 'VPH_SNBIEN'], inplace=True)

    # convert lon and lat into degree decimal
    df['lon'] = - df['LONGITUD'].apply(dms_to_dd)
    df['lat'] = df['LATITUD'].apply(dms_to_dd)

    # rename index cols
    # UPPER CASE: original variables
    # LOWER CASE: created variables
    df.rename({'ENTIDAD': 'ent', 'MUN': 'mun', 'LOC': 'loc'},
              axis='columns', inplace=True)

    if drop:
        # drop obs too close to each other
        # or obs too close to urban areas
        rural_tree = cKDTree(df.loc[df['TAM_LOC'] <= 4, ['lon', 'lat']].values)
        # the official cutoff for urban/rural is 2,500 people
        urban_tree = cKDTree(df.loc[df['TAM_LOC'] > 4, ['lon', 'lat']].values)

        # select localities with <2500 residents
        df = df.loc[df['TAM_LOC'] <= 4, :]

        rural_dist, _ = rural_tree.query(df.loc[:, ['lon', 'lat']].values, k=2)
        # excluding the point itself
        rural_dist = rural_dist[:, 1]
        urban_dist, _ = urban_tree.query(df.loc[:, ['lon', 'lat']].values)

        df = df.loc[((rural_dist > rural_cutoff) &
                     (urban_dist > urban_cutoff)), :]

    # scale VPH (household assets) variables
    for col in vph_cols:
        df[col] = df[col] / df['TVIVHAB']

    return df


def aoi_to_chip(df, indices, file_name, lon_tile_shift, lat_tile_shift):
    """Converts geo coded areas of interest to chips.

    Args:
        df (pandas.DataFrame): a DataFrame at the AOI level
            should contain columns of indices (uniquely identifying an AOI)
            and columns of ['lon', 'lat']
        indices (list of str): columns uniquely identifying an AOI
        file_name (str): pattern of file names for the chips
        lon_tile_shift, lat_tile_shift (list of int): how many chips to take
            for each AOI, e.g. [-1, 0, 1] for both will lead to 9 bordering
            chips with the lon lat of the AOI at the centroid
    Returns:
        pandas.DataFrame: a DataFrame at the chip level
    """

    # construct lon/lat jitters
    lon_tiles, lat_tiles = np.meshgrid(
        lon_tile_shift, lat_tile_shift)
    lon_tiles = lon_tiles.flatten()
    lat_tiles = lat_tiles.flatten()
    # drop other variables
    df = df.loc[:, indices + ['lon', 'lat']]
    # long format image level dataset
    df['lon_tile_width'], df['lat_tile_width'] = (
        tile_width_in_degree(df['lat'].values))
    df = pd.concat([df.assign(
        lon=lambda x: x['lon'] + x['lon_tile_width'] * lon_tile,
        lat=lambda x: x['lat'] + x['lat_tile_width'] * lat_tile,
        chip=i)
        for i, (lon_tile, lat_tile) in enumerate(zip(lon_tiles, lat_tiles))])
    # bounding box and index
    df = df.assign(
        lon_min=lambda x: x['lon'] - x['lon_tile_width'] / 2,
        lon_max=lambda x: x['lon'] + x['lon_tile_width'] / 2,
        lat_min=lambda x: x['lat'] - x['lat_tile_width'] / 2,
        lat_max=lambda x: x['lat'] + x['lat_tile_width'] / 2)
    df['index'] = df.apply(
        lambda x: file_name.format(
            *[int(x[idx]) for idx in indices + ['chip']]), axis=1)
    df.set_index('index', inplace=True, drop=True)
    df.sort_values(indices + ['chip'], inplace=True)
    return df
