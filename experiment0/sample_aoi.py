"""This script preprocesses data from Mexico 2010 CPV.

To prepare the data, from `experiment0/`, run

$ python sample_aoi.py

Then from `utils/`, run

$ ls data/GoogleStaticMap/Image | head -10
$ python download_googlestaticmap.py \
>   --log data/Experiment0/census_download_log.csv \
>   --initialize data/Experiment0/init_image.csv
$ nohup python download_googlestaticmap.py \
>   --log data/Experiment0/census_download_log.csv \
>   --num 3000 \
>   --download-dir data/GoogleStaticMap/Image \
>   > logs/download.log &
"""


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


if __name__ == '__main__':

    # define paths
    IN_DIR = 'data/CPV/Raw/ITER2010/ITER_NALDBF10.csv'
    OUT_LOC_DIR = 'data/Experiment0/census.csv'
    OUT_IMG_DIR = 'data/Experiment0/init_image.csv'

    # specify tiles to be pulled
    LON_TILE_SHIFT = [-1, 0, 1]
    LAT_TILE_SHIFT = [-1, 0, 1]

    # number of sampled localities
    N_MAIN = 200  # main sample: neither small nor large
    N_SMALL = 50  # small localities, * or N/D
    N_LARGE = 50  # large localities, TAM_LOC == 2-4

    # cutoff values for dropping observations too close to each other
    URBAN_CUTOFF = 0.1
    RURAL_CUTOFF = 0.01

    # specify data type for faster I/O
    df = pd.read_csv(IN_DIR, nrows=0)

    # select key variables and specify dtypes
    main_cols = ['ENTIDAD', 'MUN', 'LOC',
                 'LONGITUD', 'LATITUD',
                 'POBTOT', 'VIVTOT', 'TVIVHAB', 'TAM_LOC']
    vph_cols = [col for col in df.columns if col.startswith('VPH')]

    # read data frame
    # parse * (masked for privacy) or N/D (information not collected) as NA
    df = pd.read_csv(IN_DIR, usecols=main_cols + vph_cols,
                     na_values=['*', 'N/D'], dtype='Int32')

    # drop obs without coordinates (regional aggregates)
    df.dropna(subset=['LONGITUD', 'LATITUD'], inplace=True)
    # convert lon and lat into degree decimal
    df['lon'] = - df['LONGITUD'].apply(dms_to_dd)
    df['lat'] = df['LATITUD'].apply(dms_to_dd)

    # rename index cols
    # UPPER CASE: original variables
    # LOWER CASE: created variables
    df.rename({'ENTIDAD': 'ent', 'MUN': 'mun', 'LOC': 'loc'},
              axis='columns', inplace=True)

    # drop obs too close to each other
    # or obs too close to urban areas
    rural_tree = cKDTree(df.loc[df['TAM_LOC'] <= 4, ['lon', 'lat']].values)
    # the official cutoff for urban/rural is 2,500 people
    urban_tree = cKDTree(df.loc[df['TAM_LOC'] > 4, ['lon', 'lat']].values)

    # select localities with 1-249/250-2500 residents
    df = df.loc[df['TAM_LOC'] <= 4, :]

    rural_dist, _ = rural_tree.query(df.loc[:, ['lon', 'lat']].values, k=2)
    # excluding the point itself
    rural_dist = rural_dist[:, 1]
    urban_dist, _ = urban_tree.query(df.loc[:, ['lon', 'lat']].values)

    # sample localities to reduce no. of files downloaded
    df = pd.concat([
        # main sample
        (df.loc[(df['TAM_LOC'] == 1) & df['VPH_SNBIEN'].notna(), :]
         .sample(n=N_MAIN).assign(sample='main')),
        # sample of small localities
        (df.loc[df['VPH_SNBIEN'].isna(), :]
         .sample(n=N_SMALL).assign(sample='small')),
        # sample of large localities
        (df.loc[df['TAM_LOC'] > 1, :]
         .sample(n=N_LARGE).assign(sample='large'))
    ])

    # save locality level census data 
    df.to_csv(OUT_LOC_DIR, index=False)

    # construct lon/lat jitters
    lon_tile_shifts, lat_tile_shifts = np.meshgrid(
        LON_TILE_SHIFT, LAT_TILE_SHIFT)
    lon_tile_shifts = lon_tile_shifts.flatten()
    lat_tile_shifts = lat_tile_shifts.flatten()
    # drop other variables
    df = df.loc[:, ['ent', 'mun', 'loc', 'lon', 'lat']]
    # long format image level dataset
    df = pd.concat(
        [df.assign(lon_shift=lambda x: (tile_width_in_degree(x['lat'])[0] *
                                        lon_tile_shift),
                   lat_shift=lambda x: (tile_width_in_degree(x['lat'])[1] *
                                        lat_tile_shift),
                   chip=i)
         for i, (lon_tile_shift, lat_tile_shift)
         in enumerate(zip(lon_tile_shifts, lat_tile_shifts))])
    # add image centroid shifts
    df['lon'] = df['lon'] + df['lon_shift']
    df['lat'] = df['lat'] + df['lat_shift']
    # assign index
    df['index'] = df.apply(
        lambda x: 'ENT{:02d}MUN{:03d}LOC{:04d}CHIP{:02d}'
                  .format(int(x['ent']), int(x['mun']),
                          int(x['loc']), int(x['chip'])),
        axis=1)
    df.set_index('index', inplace=True, drop=True)
    df.sort_values(['ent', 'mun', 'loc', 'chip'], inplace=True)
    df.to_csv(OUT_IMG_DIR)
