import pandas as pd
import skimage.color
from sklearn.cluster import KMeans
from maskrcnn.postprocess.analysis import winsorize


IN_DIR = 'data/Siaya/Merged/sat_raw.csv'
OUT_DIR = 'data/Siaya/Merged/sat.csv'
K = 8  # number of clusters

df = pd.read_csv(IN_DIR)
# variable selection
df = df.loc[:, ['angle', 'R_mean', 'G_mean', 'B_mean',
                'area', 'centroid_lon', 'centroid_lat']]
# unit conversion, winsorize to reduce the influence of outliers
df.loc[:, 'area'] *= ((0.001716 * 111000 / 800) ** 2)  # in sq meters
df.loc[:, 'area'] = winsorize(df['area'], 1, 99)

# color grouping
rgb = df.loc[:, ['R_mean', 'G_mean', 'B_mean']].values
lab = skimage.color.rgb2lab(rgb)
m = KMeans(n_clusters=K, random_state=0)
m.fit(lab)
df.loc[:, 'color_group'] = m.labels_

df.to_csv(OUT_DIR, index=False)
