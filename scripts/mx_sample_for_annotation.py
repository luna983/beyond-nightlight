import os
import pandas as pd
import shutil


# image index data
IDX_IN_DIR = 'data/Mexico/Meta/aoi.csv'
# download log data
LOG_IN_DIR = 'data/Mexico/Meta/aoi_download_log.csv'
# census data
CEN_IN_DIR = 'data/Mexico/Meta/census.csv'

SAT_IN_IMG_DIR = 'data/Mexico/Image'
SAT_OUT_IMG_DIR = 'data/Mexico/ToSupervisely'

# read image index data frame
df_idx = pd.merge(pd.read_csv(IDX_IN_DIR),
                  pd.read_csv(LOG_IN_DIR),
                  how='outer', on='index')
df_idx = df_idx.loc[:, ['index', 'ent', 'mun', 'loc', 'chip',
                        'status', 'lon_min', 'lon_max', 'lat_min', 'lat_max']]

# read census data
df_cen = pd.read_csv(CEN_IN_DIR)

# drop NA localities with no images
df_cen = pd.merge(
    df_cen,
    df_idx.groupby(['ent', 'mun', 'loc']).agg(
        no_missing=pd.NamedAgg(column='status',
                               aggfunc=lambda x: sum(pd.isna(x)))),
    how='outer',
    on=['ent', 'mun', 'loc'])
df_cen = df_cen.loc[df_cen['no_missing'] == 0, :]

df_cen = df_cen.sample(n=20, random_state=0).loc[:, ['ent', 'mun', 'loc']].reset_index(drop=True)

df_idx = pd.merge(df_cen, df_idx, how='left', on=['ent', 'mun', 'loc'])

files = df_idx.loc[:, 'index'].tolist()

for start, folder in zip([0, 100, 200, 300, 400],
                         ['batch1', 'batch2', 'batch3', 'batch4', 'batch5']):
    os.makedirs(os.path.join(SAT_OUT_IMG_DIR, folder), exist_ok=True)
    for file in files[start:start + 100]:
        shutil.copyfile(
            os.path.join(SAT_IN_IMG_DIR, file + '.png'),
            os.path.join(SAT_OUT_IMG_DIR, folder, file + '.png'))
