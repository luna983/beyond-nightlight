import os
import glob
import tqdm
import pandas as pd
import geopandas as gpd

from maskrcnn.postprocess.polygonize import load_ann

# AOI index data w/ georeferencing info
AOI_IN_DIR = 'data/Experiment1/aoi.csv'

# satellite derived data
SAT_IN_ANN_DIR = 'data/GiveDirectly/Pred/infer/'
SAT_IN_IMG_DIR = 'data/GiveDirectly/Image/'
SAT_OUT_GEOMS_DIR = 'data/Experiment1/geoms/'
SAT_OUT_GEOM_DIR = 'data/Experiment1/sat.geojson'

# read image index data frame
df = pd.read_csv(AOI_IN_DIR)
df.loc[:, 'index'] = df['index'].str.replace('_0', '')
df.set_index('index', inplace=True)


# read sat annotations
ann_files = glob.glob(SAT_IN_ANN_DIR + '*.json')
img_files = [os.path.join(
    SAT_IN_IMG_DIR,
    (os.path.relpath(f, SAT_IN_ANN_DIR).replace('.json', '.png')))
    for f in ann_files]
for ann_file, img_file in tqdm.tqdm(zip(ann_files, img_files)):
    idx = os.path.basename(ann_file).split('.')[0]
    if idx not in df.index:
        print('skipped: ', idx)
        continue
    else:
        extent = tuple(
            df.loc[idx, ['lon_min', 'lat_min', 'lon_max', 'lat_max']].values)
    df_file = load_ann(ann_file=ann_file,
                       img_file=img_file,
                       extent=extent,
                       out_dir=SAT_OUT_GEOMS_DIR)

# collate all annotations
files = glob.glob(SAT_OUT_GEOMS_DIR + '*.geojson')
df_all = [gpd.read_file(file) for file in files]
df_all = pd.concat(df_all)
# df_all = df_all.loc[df_all.geometry.intersects(bound), :]  # drop outside geoms
df_all.to_file(SAT_OUT_GEOM_DIR, driver='GeoJSON', index=False)
