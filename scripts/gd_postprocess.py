import pandas as pd
from sklearn.cluster import KMeans


IN_DIR = 'data/Siaya/Merged/sat_raw.csv'
OUT_DIR = 'data/Siaya/Merged/sat.csv'
K = 5  # number of clusters

df = pd.read_csv(IN_DIR)
# variable selection
df = df.loc[:, ['angle', 'R_mean', 'G_mean', 'B_mean',
                'area', 'centroid_lon', 'centroid_lat']]
# unit conversion
df.loc[:, 'area'] *= ((0.001716 * 111000 / 800) ** 2)  # in sq meters

# color grouping
m = KMeans(n_clusters=K, random_state=0)
m.fit(df.loc[:, ['R_mean', 'G_mean', 'B_mean']].values)
df.loc[:, 'color_group'] = m.labels_

df.to_csv(OUT_DIR, index=False)
