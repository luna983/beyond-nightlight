from maskrcnn.preprocess.sample_aoi import load_df, aoi_to_chip


# define paths
IN_DIR = 'data/External/MexicoCPV/ITER_NALDBF10.csv'
OUT_LOC_DIR = 'data/Mexico/Meta/census.csv'
OUT_IMG_DIR = 'data/Mexico/Meta/aoi.csv'

# specify tiles to be pulled
LON_TILE_SHIFT = [-2, -1, 0, 1, 2]
LAT_TILE_SHIFT = [-2, -1, 0, 1, 2]

# number of sampled localities
N = 200
SAMPLE_NAME = '2019Oct9'

df = load_df(IN_DIR).sample(n=N, random_state=0).assign(sample=SAMPLE_NAME)
# save locality level census data
df.to_csv(OUT_LOC_DIR, index=False)
# save chip level data
df = aoi_to_chip(df=df, indices=['ent', 'mun', 'loc'],
                 file_name='ENT{:02d}MUN{:03d}LOC{:04d}CHIP{:02d}',
                 lon_tile_shift=LON_TILE_SHIFT,
                 lat_tile_shift=LAT_TILE_SHIFT)
df.to_csv(OUT_IMG_DIR)
