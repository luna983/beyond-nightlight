import os
import numpy as np
import pandas as pd


def calculate_tile_shape_in_degree(latitude, zoom_level=19,
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

    phi = latitude / 180  # in radians
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


if __name__ == '__main__':
    # define paths
    IN_DIR = 'data/CPV/Raw/ITER2010/ITER_NALDBF10.csv'
    OUT_DIR = 'data/DataFrame'

    # define longitude and latitude jitter degrees for mexico
    JIT_LON, JIT_LAT = calculate_tile_shape_in_degree(latitude=23.6345)

    # read data frame
    df = pd.read_csv(IN_DIR)

    # select localities with 1-249 residents
    df = df[df['TAM_LOC'] == 1]
    # sample localities to reduce no. of files downloaded
    df = df.sample(n=200, replace=True)

    # select key variables
    df = df.filter(items=['ENTIDAD', 'MUN', 'LOC',
                          'LONGITUD', 'LATITUD',
                          'POBTOT', 'VIVTOT', 'TVIVHAB'])
    df.columns = ['ent', 'mun', 'loc', 'lon', 'lat',
                  'pop', 'houses', 'inhabited_houses']

    # convert lon and lat into degree decimal
    df['lon'] = - df['lon'].astype('int64').apply(dms_to_dd)
    df['lat'] = df['lat'].astype('int64').apply(dms_to_dd)
    lon_shift = [-JIT_LON, 0, JIT_LON]
    lat_shift = [-JIT_LAT, 0, JIT_LAT]
    shifts = [[lon, lat] for lon in lon_shift for lat in lat_shift]
    df_image = pd.concat([df.assign(lon_shift=shift[0],
                                    lat_shift=shift[1],
                                    chip=i)
                          for i, shift in enumerate(shifts)])
    df_image['lon'] = df_image['lon'] + df_image['lon_shift']
    df_image['lat'] = df_image['lat'] + df_image['lat_shift']

    df_image['index'] = df_image.apply(
        lambda x: 'ENT{:02d}MUN{:03d}LOC{:04d}CHIP{:02d}'
                  .format(x['ent'], x['mun'], x['loc'], x['chip']),
        axis=1)
    df_image.set_index('index', inplace=True, drop=True)

    df_image.sort_values(['ent', 'mun', 'loc', 'chip'], inplace=True)

    df_image.to_csv(os.path.join(OUT_DIR, 'sampled_localities.csv'))
