"""This script preprocesses data from Mexico 2010 CPV.

To prepare the data, from `experiment0/`, run

$ python sample_aoi.py

Then from `utils/`, run

$ ls data/GoogleStaticMap/Image | head -10
$ python download_googlestaticmap.py \
>   --log data/Experiment0/census_download_log.csv \
>   --initialize data/Experiment0/census.csv
$ nohup python download_googlestaticmap.py \
>   --log data/Experiment0/census_download_log.csv \
>   --num 3000 \
>   --download-dir data/GoogleStaticMap/Image \
>   > logs/download.log &
"""


import numpy as np
import pandas as pd


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


if __name__ == '__main__':

    # define paths
    IN_DIR = 'data/CPV/Raw/ITER2010/ITER_NALDBF10.csv'
    OUT_DIR = 'data/Experiment0/census.csv'

    # specify tiles to be pulled
    LON_TILE_SHIFT = [-1, 0, 1]
    LAT_TILE_SHIFT = [-1, 0, 1]

    # number of sampled localities
    N = 200

    # read data frame
    df = pd.read_csv(IN_DIR)

    # select localities with 1-249 residents
    df = df[df['TAM_LOC'] == 1]
    # sample localities to reduce no. of files downloaded
    df = df.sample(n=N, replace=True)

    # select key variables
    df = df.filter(items=['ENTIDAD', 'MUN', 'LOC',
                          'LONGITUD', 'LATITUD',
                          'POBTOT', 'VIVTOT', 'TVIVHAB'])
    df.columns = ['ent', 'mun', 'loc', 'lon', 'lat',
                  'pop', 'houses', 'inhabited_houses']
    df.dropna(how='any', inplace=True)

    # convert lon and lat into degree decimal
    df['lon'] = - df['lon'].astype('int64').apply(dms_to_dd)
    df['lat'] = df['lat'].astype('int64').apply(dms_to_dd)

    # construct lon/lat jitters
    lon_tile_shifts, lat_tile_shifts = np.meshgrid(
        LON_TILE_SHIFT, LAT_TILE_SHIFT)
    lon_tile_shifts = lon_tile_shifts.flatten()
    lat_tile_shifts = lat_tile_shifts.flatten()

    df_image = pd.concat(
        [df.assign(lon_shift=lambda x: (tile_width_in_degree(x['lat'])[0] *
                                        lon_tile_shift),
                   lat_shift=lambda x: (tile_width_in_degree(x['lat'])[1] *
                                        lat_tile_shift),
                   chip=i)
         for i, (lon_tile_shift, lat_tile_shift)
         in enumerate(zip(lon_tile_shifts, lat_tile_shifts))])

    df_image['lon'] = df_image['lon'] + df_image['lon_shift']
    df_image['lat'] = df_image['lat'] + df_image['lat_shift']

    df_image['index'] = df_image.apply(
        lambda x: 'ENT{:02d}MUN{:03d}LOC{:04d}CHIP{:02d}'
                  .format(int(x['ent']), int(x['mun']),
                          int(x['loc']), int(x['chip'])),
        axis=1)
    df_image.set_index('index', inplace=True, drop=True)
    df_image.sort_values(['ent', 'mun', 'loc', 'chip'], inplace=True)
    df_image.to_csv(OUT_DIR)
