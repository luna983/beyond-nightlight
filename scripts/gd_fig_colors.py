import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='ticks', font='Helvetica', font_scale=1)

IN_DIR = 'data/Siaya/Merged/sat.csv'
OUT_DIR = 'output/fig-colors/raw.pdf'
FRAC_RANDOM = 0.0004
COLUMN_WIDTH = 15

df = pd.read_csv(IN_DIR)

n_clusters = df['color_group'].max() + 1

separator = np.full((1, COLUMN_WIDTH + 2, 3), 1.)
result = separator.copy()

fig, ax = plt.subplots(figsize=(9, 8))
ax.axis('off')
for i, df_group in df.groupby('color_group'):
    random = df_group.sample(
        frac=FRAC_RANDOM,
    ).loc[:, ['R_mean', 'G_mean', 'B_mean']].values / 255
    pad_width = (
        -(-random.shape[0] // COLUMN_WIDTH)
        * COLUMN_WIDTH - random.shape[0])
    random = np.pad(random, ((0, pad_width), (0, 0)),
                    mode='constant', constant_values=1)
    random = random.reshape((-1, COLUMN_WIDTH, 3))
    mean = (df_group.loc[:, ['R_mean', 'G_mean', 'B_mean']]
                    .mean(axis=0).values / 255)
    pad_width = random.shape[0] * 2 - 1
    mean = np.pad(mean[np.newaxis, :], ((0, pad_width), (0, 0)),
                  mode='constant', constant_values=1)
    mean = mean.reshape((-1, 2, 3))
    block = np.concatenate((mean, random), axis=1)
    result = np.concatenate(
        (result, block), axis=0)
    result = np.concatenate(
        (result, separator), axis=0)
ax.imshow(result)
fig.savefig(OUT_DIR)
