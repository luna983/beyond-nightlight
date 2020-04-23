import os
import tempfile
import glob
import tqdm
import pandas as pd
import geopandas as gpd

from maskrcnn.postprocess.polygonize import load_ann


# AOI index data w/ georeferencing info
AOI_IN_DIR = 'data/Siaya/Meta/aoi.csv'
# download log data
LOG_IN_DIR = 'data/Siaya/Meta/aoi_download_log.csv'

# satellite derived data
SAT_IN_ANN_DIR = 'data/Siaya/Pred/infer/'
SAT_IN_IMG_DIR = 'data/Siaya/Image/'
SAT_OUT_GEOM_DIR = 'data/Siaya/Merged/sat.geojson'

# boundary
BOUND_IN_DIR = 'data/External/GiveDirectly/figure2/SampleArea.shp'

# read boundary shapefile
bound, = gpd.read_file(BOUND_IN_DIR)['geometry']

# read image index data frame
df = pd.merge(pd.read_csv(AOI_IN_DIR),
              pd.read_csv(LOG_IN_DIR).loc[:, 'index'],
              how='right', on='index')
# quick drop
df = df.loc[df['lat_max'] >= bound.bounds[1]]
df.set_index('index', inplace=True)

# read sat annotations
ann_files = glob.glob(SAT_IN_ANN_DIR + '*.json')
img_files = [os.path.join(
    SAT_IN_IMG_DIR,
    (os.path.relpath(f, SAT_IN_ANN_DIR).replace('.json', '.png')))
    for f in ann_files]

with tempfile.TemporaryDirectory() as tmp_dir:
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
                           out_dir=tmp_dir)

    # collate all annotations
    files = glob.glob(os.path.join(tmp_dir, '*.geojson'))
    df_all = [gpd.read_file(file) for file in files]
    df_all = pd.concat(df_all)
    # df_all = df_all.loc[df_all.geometry.intersects(bound), :]  # drop outside geoms
    df_all.to_file(SAT_OUT_GEOM_DIR, driver='GeoJSON', index=False)
