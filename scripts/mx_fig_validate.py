import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from skmisc.loess import loess

from maskrcnn.postprocess.analysis import winsorize, load_nightlight_from_point


matplotlib.rc('pdf', fonttype=42)
sns.set(style='ticks', font='Helvetica')

plt.rc('font', size=11)  # controls default text sizes
plt.rc('axes', titlesize=11)  # fontsize of the axes title
plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
plt.rc('legend', fontsize=11)  # legend fontsize
plt.rc('figure', titlesize=11)  # fontsize of the figure title


def plot_scatter(col_x_key, col_y_key, col_x_label, col_y_label, df, ax,
                 transform_x=lambda x: x, transform_y=lambda x: x,
                 xlim=None, ylim=None, xticks=None, yticks=None,
                 xticklabels=None, yticklabels=None,
                 alpha=0.5, line=False, frac=0.3,
                 square=False):
    """Generates scatter plots w/ correlation coef.

    Args:
        col_x_key, col_y_key (str): keys for variables
        col_x_label, col_y_label (str): labels for axes
        df (pandas.DataFrame): stores the data
        out_dir (str): output directory
        transform_x, transform_y (function): how the two axes should be
            transformed
        xlim, ylim (tuple of float): limits of the transformed axes
        xticks, yticks, xticklabels, yticklabels (list)
        figsize (tuple of float): passed to plt.subplots()
        alpha (float): transparency
        line (bool): whether a non parametric fit line should be displayed
        frac (float): passed to lowess() for smoothing
        square (bool): whether x and y axis should be the same and
            whether a 45 degree line should be plotted
    """

    df_nona = (df.loc[:, [col_x_key, col_y_key]]
                 .dropna().sort_values(by=col_x_key).astype('float'))
    cols = df_nona.values
    corr, _ = pearsonr(cols[:, 0], cols[:, 1])
    x = transform_x(cols[:, 0])
    y = transform_y(cols[:, 1])
    ax.plot(x, y,
            markeredgecolor='none',
            marker='o', color='dimgrey',
            linestyle='None', alpha=alpha,
            markersize=4)
    if line:
        m = loess(x, y)
        m.fit()
        pred = m.predict(x, stderror=True).confidence()
        ax.plot(x, pred.fit,
                color='dimgray', linewidth=1, alpha=0.4)
        ax.fill_between(x, pred.lower, pred.upper,
                        color='dimgray', alpha=.2)
    if square:
        ax.axis('square')
        ax.plot(xlim, ylim, '--', color='gray', linewidth=2)
    ax.set_title(f'{col_y_label}\nCorrelation: {corr:.2f}',
                 loc='left')
    ax.set_xlabel(col_x_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.spines['left'].set_bounds(ax.get_yticks()[0], ax.get_yticks()[-1])
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x', colors='dimgray')
    ax.tick_params(axis='y', colors='dimgray')
    ax.grid(False)


def load_index(IDX_IN_DIR, LOG_IN_DIR):
    # LOAD IMAGE META DATA
    # read image index data frame
    df_idx = pd.merge(
        pd.read_csv(IDX_IN_DIR), pd.read_csv(LOG_IN_DIR),
        how='outer', on='index')
    df_idx = df_idx.loc[:, [
        'index', 'ent', 'mun', 'loc', 'chip', 'status',
        'lon_min', 'lon_max', 'lat_min', 'lat_max']]
    return df_idx


def load_census(df_idx, CEN_IN_DIR):
    # LOAD CENSUS DATA
    # read census data
    df_cen = pd.read_csv(CEN_IN_DIR)

    # select sample
    df_cen = df_cen.loc[df_cen['sample'] == SAMPLE_NAME, :]

    # drop NA localities with no images
    df_cen = pd.merge(
        df_cen,
        df_idx.groupby(['ent', 'mun', 'loc']).agg(
            no_missing=pd.NamedAgg(
                column='status', aggfunc=lambda x: sum(pd.isna(x)))),
        how='outer',
        on=['ent', 'mun', 'loc'])
    df_cen = df_cen.loc[df_cen['no_missing'] == 0, :]

    # calculate PCA scores
    db_cols = ['VPH_RADIO', 'VPH_TV', 'VPH_REFRI', 'VPH_LAVAD',
               'VPH_AUTOM', 'VPH_PC', 'VPH_TELEF', 'VPH_CEL', 'VPH_INTER']
    housing_cols = ['VPH_PISODT', 'VPH_2YMASD', 'VPH_3YMASC']
    public_cols = ['VPH_C_ELEC', 'VPH_AGUADV', 'VPH_EXCSA', 'VPH_DRENAJ']
    vph_cols = db_cols + housing_cols + public_cols  # all

    for cols, output_col in zip(
            [db_cols, housing_cols, public_cols, vph_cols],
            ['cen_durable_score_pca', 'cen_housing_score_pca',
             'cen_public_score_pca', 'cen_asset_score_pca']):
        # normalize before feeding into PCA
        raw = df_cen.loc[:, cols].values
        normalized = ((raw - raw.mean(axis=0)[np.newaxis, :])
                      / (raw.std(axis=0)[np.newaxis, :] + 1e-10))
        m = PCA(n_components=1)
        pca_score = m.fit_transform(normalized)
        df_cen.loc[:, output_col] = ((pca_score - pca_score.mean())
                                     / (pca_score.std() + 1e-10))

    df_cen['cen_housing_score_pca'] *= -1
    df_cen['cen_public_score_pca'] *= -1

    # renamed columns in census, translate to English
    CEN_COLS = {
        'POBTOT': 'cen_pop',
        'VIVTOT': 'cen_house',
        'TVIVHAB': 'cen_inhab',
        # 'VPH_SNBIEN': 'cen_nodurable',
        # 'VPH_1CUART': 'cen_1room',
        # 'VPH_3YMASC': 'cen_3plusroom',
        # 'VPH_REFRI': 'cen_refri',
        # 'VPH_AUTOM': 'cen_autom',
        # 'VPH_C_ELEC': 'cen_elec',
        # 'VPH_PISODT': 'cen_floor',
        # 'VPH_AGUADV': 'cen_water',
        # 'VPH_EXCSA': 'cen_toilet',
        # 'VPH_DRENAJ': 'cen_drainage',
        # 'VPH_INTER': 'cen_internet',
    }

    # rename columns
    df_cen = df_cen.rename(columns=CEN_COLS)
    print('No. of localities in the sample: ', df_cen.shape[0])
    return df_cen


def load_satellite(df_idx, SAT_IN_DIR):
    # read satellite predictions
    df_sat = pd.read_csv(SAT_IN_DIR)
    # link to locality identifier
    df_sat = pd.merge(df_sat, df_idx, how='left', on='index')
    # create new var
    df_sat.loc[:, 'RGB_mean'] = (df_sat.loc[:, ['R_mean', 'G_mean', 'B_mean']]
                                       .mean(axis=1))

    # grouping into localities
    df_sat = df_sat.groupby(['ent', 'mun', 'loc']).agg(
        sat_house=pd.NamedAgg(column='area', aggfunc='count'),
        sat_size_mean=pd.NamedAgg(column='area', aggfunc=np.nanmean),
        sat_size_sum=pd.NamedAgg(column='area', aggfunc=np.nansum),
        sat_lum_mean=pd.NamedAgg(column='RGB_mean', aggfunc=np.nanmean),
    )

    # scale areas / distances
    df_sat[['sat_size_mean', 'sat_size_sum']] *= (
        ((0.001716 * 111000 / 800) ** 2) *
        np.cos(23 / 180 * np.pi))  # in sq meters
    # winsorize
    df_sat.loc[:, 'sat_size_sum_wins'] = winsorize(
        df_sat['sat_size_sum'], 0, 99)
    df_sat.loc[:, 'sat_size_mean_wins'] = winsorize(
        df_sat['sat_size_mean'], 0, 99)
    return df_sat


if __name__ == '__main__':

    # input path
    # image index data
    IDX_IN_DIR = 'data/Mexico/Meta/aoi.csv'
    # download log data
    LOG_IN_DIR = 'data/Mexico/Meta/aoi_download_log.csv'
    # census data
    CEN_IN_DIR = 'data/Mexico/Meta/census.csv'
    # satellite data
    NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_MX_2019.tif'
    SAT_IN_DIR = 'data/Mexico/Merged/sat.csv'
    # output path
    OUT_DIR = 'output/fig-mx'
    OUT_RAW_DATA_DIR = 'fig_raw_data'
    # sample selection
    SAMPLE_NAME = '2019Oct9'

    # load data
    df_idx = load_index(IDX_IN_DIR, LOG_IN_DIR)
    df_sat = load_satellite(df_idx, SAT_IN_DIR)
    df_cen = load_census(df_idx, CEN_IN_DIR)
    # merge satellite and census
    df = pd.merge(
        df_sat,
        df_cen,
        how='right', on=['ent', 'mun', 'loc'])
    df.loc[:, 'sat_house'] = df['sat_house'].fillna(0)
    df.loc[:, 'sat_size_sum'] = df['sat_size_sum'].fillna(0)
    # compute per capita values
    df.loc[:, 'sat_size_sum_pc'] = df['sat_size_sum'] / df['cen_pop']
    # winsorize
    df.loc[:, 'sat_size_sum_pc_wins'] = winsorize(df['sat_size_sum_pc'], 0, 99)
    # load nightlight values
    df = load_nightlight_from_point(df, NL_IN_DIR)
    df = df.rename({'nightlight': 'sat_nightlight'}, axis=1)
    # plotting begins
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    plot_scatter(
        col_x_key='cen_pop',
        col_x_label='Census: Population Count',
        transform_x=lambda x: np.log10(x + 1),
        xlim=(np.log10(2 + 1), np.log10(3000 + 1)),
        xticks=[np.log10(10 + 1), np.log10(100 + 1), np.log10(1000 + 1)],
        xticklabels=[10, 100, 1000],
        col_y_key='sat_house',
        col_y_label='Satellite: Number of Houses',
        transform_y=lambda x: np.log10(x + 1),
        ylim=(np.log10(0.7), np.log10(1500 + 1)),
        yticks=[np.log10(1 + 1), np.log10(10 + 1),
                np.log10(100 + 1), np.log10(1000 + 1)],
        yticklabels=[1, 10, 100, 1000],
        line=True, df=df,
        ax=axes[1])

    # plot_scatter(
    #     col_x_key='cen_asset_score_pca',
    #     col_x_label='Census: Asset Score',
    #     xlim=(-1.6, 1.6),
    #     xticks=[-1, 0, 1],
    #     col_y_key='sat_size_mean',
    #     col_y_label='Satellite: Average House Size (m2)',
    #     ylim=(-1, 220),
    #     yticks=[0, 100, 200],
    #     line=True, df=df,
    #     ax=axes[1, 1])

    # plot_scatter(
    #     col_x_key='cen_durable_score_pca',
    #     col_x_label='Census: Durable Asset Score',
    #     xlim=(-1.25, 1.25),
    #     xticks=[-1, 0, 1],
    #     col_y_key='sat_size_mean',
    #     col_y_label='Satellite: Average House Size (m2)',
    #     ylim=(-1, 220),
    #     yticks=[0, 100, 200],
    #     line=True, df=df,
    #     ax=axes[2, 1])

    plot_scatter(
        col_x_key='cen_pop',
        col_x_label='Census: Population Count',
        transform_x=lambda x: np.log10(x + 1),
        xlim=(np.log10(3), np.log10(3000)),
        xticks=[np.log10(10 + 1), np.log10(100 + 1), np.log10(1000 + 1)],
        xticklabels=[10, 100, 1000],
        col_y_key='sat_nightlight',
        col_y_label='Satellite: Night Light',
        ylim=(-0.01, 4),
        yticks=[0, 2, 4],
        alpha=0.3, line=True, df=df,
        ax=axes[0])

    # plot_scatter(
    #     col_x_key='cen_asset_score_pca',
    #     col_x_label='Census: Asset Score',
    #     xlim=(-1.75, 1.75),
    #     xticks=[-1, 0, 1],
    #     col_y_key='sat_nightlight',
    #     col_y_label='Satellite: Night Light',
    #     ylim=(-0.01, 4),
    #     yticks=[0, 2, 4],
    #     alpha=0.3,
    #     line=True, df=df,
    #     ax=axes[1, 0])

    # plot_scatter(
    #     col_x_key='cen_durable_score_pca',
    #     col_x_label='Census: Durable Score',
    #     xlim=(-1.5, 1.5),
    #     xticks=[-1, 0, 1],
    #     col_y_key='sat_nightlight',
    #     col_y_label='Satellite: Night Light',
    #     ylim=(-0.01, 4),
    #     yticks=[0, 2, 4],
    #     alpha=0.3,
    #     line=True, df=df,
    #     ax=axes[2, 0])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'raw.pdf'))

    # massive plotting begins
    # sat_cols = [col for col in df.columns if col.startswith('sat')]
    # cen_cols = [col for col in df.columns if col.startswith('cen')]
    # for f in glob.glob(os.path.join(OUT_DIR, 'archive/*.pdf')):
    #     os.remove(f)
    # for sat_col in sat_cols:
    #     for cen_col in cen_cols:
    #         plot_scatter(col_x_key=cen_col, col_y_key=sat_col,
    #                      col_x_label=cen_col, col_y_label=sat_col,
    #                      line=True,
    #                      df=df, out_dir=os.path.join(OUT_DIR, 'archive'))

    pd.DataFrame({
        'census_population_count': df['cen_pop'].values,
        'satellite_night_light': df['sat_nightlight'].values,
        'satellite_number_of_houses': df['sat_house'].astype(int).values,
    }).to_csv(
        os.path.join(OUT_RAW_DATA_DIR, 'fig-mx.csv'), index=False)
