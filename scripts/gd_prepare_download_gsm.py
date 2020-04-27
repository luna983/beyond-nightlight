import numpy as np
import scipy.spatial
import pandas as pd
import shapely
import shapely.geometry
import geopandas as gpd
import matplotlib.pyplot as plt

from maskrcnn.preprocess.sample_aoi import aoi_to_chip


IN_DIR_SHP = 'data/External/GiveDirectly/adm_shapefile/gadm36_KEN_1.shp'
LAKE_SHP = 'data/External/GiveDirectly/adm_shapefile/KEN_Lakes.shp'
OUT_IMG_DIR = 'data/Siaya/Meta/aoi.csv'

# read shapefiles
df_shp = gpd.read_file(IN_DIR_SHP)
shp, = df_shp.loc[df_shp['NAME_1'] == 'Siaya', 'geometry'].values

# remove lake
lake = gpd.read_file(LAKE_SHP)
lake, = lake.loc[lake['LAKE'] == 'Lake Victoria', 'geometry'].values
shp = shp.difference(lake)

df_shp = gpd.GeoDataFrame({'COUNTY': [41]}, geometry=[shp])

# save chip level data
df_chip = aoi_to_chip(df=df_shp, indices=['COUNTY'],
                      file_name='COUNTY{}CHIP{:08d}',
                      input_type='polygon')
df_chip.to_csv(OUT_IMG_DIR)
