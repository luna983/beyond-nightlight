from maskrcnn.preprocess.sample_aoi import aoi_to_chip

# define paths
SVY_IN_DIR = 'data/GiveDirectly/Survey/GE_Luna_Extract_2018-09-19.dta'
OUT_IMG_DIR = 'data/Experiment1/aoi.csv'


# specify tiles to be pulled
LON_TILE_SHIFT = [0]
LAT_TILE_SHIFT = [0]

df = pd.read_stata(SVY_IN_DIR)
df = df.dropna(subset=['s19_gps_latitude', 's19_gps_longitude'])

# save chip level data
df = aoi_to_chip(df=df, indices=['ent', 'mun', 'loc'],
                 file_name='ENT{:02d}MUN{:03d}LOC{:04d}CHIP{:02d}',
                 lon_tile_shift=LON_TILE_SHIFT,
                 lat_tile_shift=LAT_TILE_SHIFT)
df.to_csv(OUT_IMG_DIR)
