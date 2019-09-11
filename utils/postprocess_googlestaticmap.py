import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter(col_x, col_y):
    """Plots col_y against col_x."""
    df.plot.scatter(x=col_x, y=col_y)
    plt.savefig(os.path.join(OUT_DIR, '{}_vs_{}.pdf'.format(col_x, col_y)))
    plt.close('all')


if __name__ == '__main__':
    # census data
    CEN_IN_DIR = 'data/DataFrame/sampled_localities.csv'
    # satellite derived data
    SAT_IN_DIR = 'data/GoogleStaticMap/Pred/infer/annotations_pred.json'
    # output path
    OUT_DIR = 'data/GoogleStaticMap/output/'
    # read data frame
    df_cen = pd.read_csv(CEN_IN_DIR)
    df_image_id = df_cen[['ent', 'mun', 'loc', 'index']]
    df_cen = df_cen[['ent', 'mun', 'loc',
                     'pop', 'houses', 'inhabited_houses']]
    df_cen = df_cen.drop_duplicates()
    # read json annotations
    with open(SAT_IN_DIR, 'r') as f:
        df_sat = json.load(f)
    df_sat = [[ins['image_id_str'], ins['category_id'],
               ins['score'], ins['area']]
              for ins in df_sat]
    df_sat = pd.DataFrame(
        df_sat, columns=['index', 'category_id', 'score', 'area'])
    df_sat = pd.merge(df_sat, df_image_id, on='index', how='left')
    df_sat = df_sat.groupby(['ent', 'mun', 'loc']).agg(
        house_count=pd.NamedAgg(column='area', aggfunc='count'),
        house_area_sum=pd.NamedAgg(column='area', aggfunc=np.sum),
        house_area_mean=pd.NamedAgg(column='area', aggfunc=np.mean),
        house_area_std=pd.NamedAgg(column='area', aggfunc=np.std),
        house_area_25p=pd.NamedAgg(
            column='area',
            aggfunc=lambda x: np.percentile(x, 25)),
        house_area_median=pd.NamedAgg(column='area', aggfunc=np.median),
        house_area_75p=pd.NamedAgg(
            column='area',
            aggfunc=lambda x: np.percentile(x, 75)))
    df = pd.merge(
        df_sat, df_cen, how='inner', on=['ent', 'mun', 'loc'])
    # plotting begins
    plot_scatter('houses', 'house_count')
    plot_scatter('inhabited_houses', 'house_count')
    plot_scatter('pop', 'house_count')
