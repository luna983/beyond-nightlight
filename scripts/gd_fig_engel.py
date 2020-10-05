import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial
from skmisc.loess import loess

from maskrcnn.postprocess.analysis import (
    winsorize, load_nightlight_from_point,
    load_building, load_gd_census)


np.random.seed(0)
sns.set(style='ticks', font='Helvetica', font_scale=1)


def reg(df, y, x):
    df_nona = df.dropna(subset=[x, y])
    print(f'y: {y}, x: {x}, N = {df_nona.shape[0]}')
    results = smf.ols(y + ' ~ ' + x, data=df_nona).fit()
    beta = results.params[x]
    se = results.bse[x]
    return beta, se


def compute_est(y_coef, y_se, scale, scale_se):
    # calculate estimated effect
    est = y_coef / scale
    est_se = np.sqrt((y_se / y_coef) ** 2 + (scale_se / scale) ** 2) * abs(est)
    return est, est_se


def plot_curve(ax, method, x_col, y_col, color='dimgrey', se=True, **kwargs):
    if method == 'loess':
        m = loess(x_col, y_col, **kwargs)
        m.fit()
        pred = m.predict(x_col, stderror=True).confidence()
        pred_fit = pred.fit
        pred_lower, pred_upper = pred.lower, pred.upper
    elif method == 'linear':
        X = sm.add_constant(x_col)
        m = sm.OLS(y_col, X).fit()
        pred = m.get_prediction(X)
        pred_fit = pred.predicted_mean
        pred_lower, pred_upper = pred.conf_int().T
    else:
        raise NotImplementedError
    ax.plot(x_col, pred_fit,
            color=color, linewidth=1, alpha=0.4)
    if se:
        ax.fill_between(x_col, pred_lower, pred_upper,
                        color=color, alpha=.2)


def plot_engel(df, y, x, treat='treat',
               method='linear',
               cmap=None,
               x_label=None, y_label=None,
               x_ticks=None, x_ticklabels=None,
               y_ticks=None, y_ticklabels=None):

    # make figure
    fig, ax = plt.subplots(figsize=(4, 3))
    df_nona = df.dropna(subset=[y, x]).sort_values(by=[x])
    x_col = df_nona[x].values
    y_col = df_nona[y].values
    plot_curve(ax=ax, method=method, x_col=x_col, y_col=y_col,
               color='dimgrey', se=True)
    for color_key, df_group in df_nona.groupby(treat):
        color = cmap[color_key]
        x_col = df_group[x].values
        y_col = df_group[y].values
        plot_curve(ax=ax, method=method, x_col=x_col, y_col=y_col,
                   color=color, se=False)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.spines['left'].set_bounds(ax.get_yticks()[0], ax.get_yticks()[-1])
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x', colors='dimgray')
    ax.tick_params(axis='y', colors='dimgray')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{y}-{x}.pdf'))


def plot_est(y, labels, betas, ses, xticks=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.errorbar(
        x=betas, y=range(len(betas)),
        xerr=1.96 * np.array(ses), color='#999999',
        capsize=3, fmt='o')
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, len(betas) - 0.5)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x', colors='dimgray')
    ax.tick_params(axis='y', color='none')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'est_{y}.pdf'))


def load_survey(SVY_IN_DIR):
    # load survey data
    df_svy = pd.read_stata(SVY_IN_DIR)
    print('Observations in raw data: ', df_svy.shape[0])

    # drop households without geo coords
    df_svy = df_svy.dropna(
        subset=['latitude', 'longitude'],
    ).reset_index(drop=True)
    print('Observations w/ coords: ', df_svy.shape[0])

    # calculate per capita consumption / assets
    # convert to USD PPP by dividing by 46.5, per the GiveDirectly paper
    df_svy.loc[:, 'p2_consumption_wins_pc'] = (
        df_svy['p2_consumption_wins'].values /
        df_svy['hhsize1_BL'].values /
        46.5)
    df_svy.loc[:, 'p1_assets_pc'] = (
        df_svy['p1_assets'].values /
        df_svy['hhsize1_BL'].values /
        46.5)
    df_svy.loc[:, 'h1_10_housevalue_pc'] = (
        df_svy['h1_10_housevalue_wins_PPP'].values /
        df_svy['hhsize1_BL'].values
    )
    df_svy.loc[:, 'h1_11_landvalue_pc'] = (
        df_svy['h1_11_landvalue_wins_PPP'].values /
        df_svy['hhsize1_BL'].values
    )
    df_svy.loc[:, 'assets_house_pc'] = (
        (df_svy['p1_assets'].values + df_svy['h1_10_housevalue'].values) /
        df_svy['hhsize1_BL'].values /
        46.5)
    df_svy.loc[:, 'assets_all_pc'] = (
        ((df_svy['p1_assets'].values / 46.5) +
         df_svy['h1_11_landvalue_wins_PPP'].values +
         df_svy['h1_10_housevalue_wins_PPP'].values) /
        df_svy['hhsize1_BL'].values)

    # log and winsorize more
    df_svy.loc[:, 'logwins_p2_consumption_wins_pc'] = winsorize(
        df_svy['p2_consumption_wins_pc'], 2.5, 97.5
    ).apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )
    df_svy.loc[:, 'logwins_p1_assets_pc'] = winsorize(
        df_svy['p1_assets_pc'], 2.5, 97.5
    ).apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )
    df_svy.loc[:, 'logwins_assets_all_pc'] = winsorize(
        df_svy['assets_all_pc'], 2.5, 97.5
    ).apply(
        lambda x: np.log(x) if x > 0 else np.nan
    )

    # check missing
    assert (df_svy.loc[:, ['treat', 'hi_sat', 's1_hhid_key', 'satcluster']]
                  .notna().all().all())

    # subset to eligible sample
    # df_svy = (df_svy.loc[df_svy['h1_6_nonthatchedroof_BL'] == 0, :]
    #                 .reset_index(drop=True).copy())
    print('Observations in final sample: ', df_svy.shape[0])

    return df_svy


def match(
    df_cen, df_svy, df_sat,
    radius=0.00045,  # = 50m
):
    df_cen = df_cen.reset_index(drop=True)
    df_cen.loc[:, 'census_id'] = df_cen.index
    tree = scipy.spatial.cKDTree(
        df_cen.loc[:, ['longitude', 'latitude']].values)
    # match structures to households
    dists, cen_idxes = tree.query(
        df_sat.loc[:, ['centroid_lon', 'centroid_lat']].values, k=1)
    df_sat.loc[:, 'dist'] = dists
    df_sat.loc[:, 'census_id'] = cen_idxes
    df_sat = df_sat.loc[df_sat['dist'] < radius, :]
    # take all the structures within the radius
    df_sat = df_sat.groupby('census_id').agg(
        house_count=pd.NamedAgg(column='area', aggfunc='count'),
        area_sum=pd.NamedAgg(column='area', aggfunc='sum'),
        color_tin=pd.NamedAgg(column='color_tin', aggfunc='sum'),
        color_thatched=pd.NamedAgg(column='color_thatched', aggfunc='sum'),
        color_tin_area=pd.NamedAgg(column='color_tin_area', aggfunc='sum'),
        color_thatched_area=pd.NamedAgg(
            column='color_thatched_area', aggfunc='sum'),
    ).reset_index()
    # match surveys to households
    dists, cen_idxes = tree.query(
        df_svy.loc[:, ['longitude', 'latitude']].values, k=1)
    df_svy.loc[:, 'dist'] = dists
    df_svy.loc[:, 'census_id'] = cen_idxes
    df_svy = df_svy.loc[df_svy['dist'] < radius, :]
    df_svy = df_svy.sort_values(by=['census_id', 'dist'])
    df_svy = df_svy.drop_duplicates(subset=['census_id'], keep='first')
    # merge
    df = pd.merge(
        df_cen.loc[:, ['census_id', 'treat', 'eligible',
                       'longitude', 'latitude']],
        df_sat.loc[:, ['census_id', 'house_count', 'area_sum',
                       'color_tin', 'color_thatched',
                       'color_tin_area', 'color_thatched_area']],
        how='left', on='census_id',
    )
    df.fillna(
        {'house_count': 0, 'area_sum': 0, 'color_tin': 0,
         'color_thatched': 0, 'color_tin_area': 0, 'color_thatched_area': 0},
        inplace=True)
    df = pd.merge(
        df,
        df_svy.loc[:, ['census_id', 's1_hhid_key',
                       'hhsize1_BL', 'logwins_p2_consumption_wins_pc',
                       'logwins_assets_all_pc']],
        how='left', on='census_id',
    )

    df.loc[:, 'area_sum_pc'] = (
        df['area_sum'].values / df['hhsize1_BL'].values)
    df.loc[:, 'color_tin_area_pc'] = (
        df['color_tin_area'].values /
        df['hhsize1_BL'].values)
    df.loc[:, 'log1_area_sum_pc'] = df['area_sum_pc'].apply(
        lambda x: np.log(x + 1) if x > 0 else np.nan
    )
    df.loc[:, 'log1_color_tin_area_pc'] = (
        df['color_tin_area_pc'].apply(
            lambda x: np.log(x + 1) if x > 0 else np.nan
        )
    )
    df.loc[:, 'log1_area_sum'] = df['area_sum'].apply(
        lambda x: np.log(x + 1) if x > 0 else np.nan
    )
    df.loc[:, 'log1_color_tin_area'] = (
        df['color_tin_area'].apply(
            lambda x: np.log(x + 1) if x > 0 else np.nan
        )
    )
    print(df.describe().T)
    return df


if __name__ == '__main__':

    palette = ['#820000', '#ea0000', '#fff4da', '#5d92c4', '#070490']
    cmap = {0: palette[-1], 1: palette[0]}

    SVY_IN_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
    SAT_IN_DIR = 'data/Siaya/Merged/sat.csv'
    NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'
    IN_CENSUS_GPS_DIR = ('data/External/GiveDirectly/'
                         'GE_HH_Census_2017-07-17_cleanGPS.csv')
    IN_CENSUS_MASTER_DIR = (
        'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
    OUT_DIR = 'output/fig-engel'

    # load data
    df_sat = load_building(SAT_IN_DIR, grid=None, agg=False)
    df_svy = load_survey(SVY_IN_DIR)
    df_cen = load_gd_census(
        GPS_FILE=IN_CENSUS_GPS_DIR, MASTER_FILE=IN_CENSUS_MASTER_DIR)

    # match
    df = match(df_cen, df_svy, df_sat)
    df.loc[:, 'treat'] = df['treat'].astype(float)

    # load nightlight
    df = load_nightlight_from_point(
        df, NL_IN_DIR,
        lon_col='longitude', lat_col='latitude')

    # plotting begins
    ys = ['sat_nightlight_winsnorm',
          'log1_area_sum_pc',
          'log1_color_tin_area_pc']
    y_labels = ['Normalized nightlight values',
                'Building footprint per capita (sq meters)',
                'Tin-roof area per capita (sq meters)']
    y_ticks = [[-1, 0, 1],
               np.log(np.array([10, 30, 50, 70]) + 1),
               np.log(np.array([10, 30, 50, 70]) + 1)]
    y_ticklabels = [None,
                    [10, 30, 50, 70],
                    [10, 30, 50, 70]]
    xs = ['logwins_assets_all_pc',
          'logwins_p2_consumption_wins_pc']
    x_labels = ['Assets per capita (USD PPP)',
                'Consumption per capita (USD PPP)']
    x_ticks = [np.log([300, 1000, 3000, 8000]),
               np.log([100, 300, 1000, 3000])]
    x_ticklabels = [[300, 1000, 3000, 8000],
                    [100, 300, 1000, 3000]]

    for x, x_label, x_tick, x_ticklabel in zip(
        xs, x_labels, x_ticks, x_ticklabels
    ):
        est_labels = []
        est_betas = []
        est_ses = []
        obs, obs_se = reg(df.loc[df['eligible'] == 1, :], x, 'treat')
        est_labels.append('Observed')
        est_betas.append(obs)
        est_ses.append(obs_se)
        for y, y_label, y_tick, y_ticklabel in zip(
            ys, y_labels, y_ticks, y_ticklabels
        ):
            plot_engel(
                df=df,
                y=y,
                y_ticks=y_tick,
                y_ticklabels=y_ticklabel,
                y_label=y_label,
                x=x,
                x_ticks=x_tick,
                x_ticklabels=x_ticklabel,
                x_label=x_label,
                cmap=cmap,
            )
            y_coef, y_se = reg(df.loc[df['eligible'] == 1, :], y, 'treat')
            scale, scale_se = reg(df, y, x)
            est, est_se = compute_est(y_coef, y_se, scale, scale_se)
            print(f'y: {y}, x: {x}')
            print(f'{est:.3f}, {est_se:.3f} = '
                  f'compute_est({y_coef:.3f}, {y_se:.3f}, '
                  f'{scale:.3f}, {scale_se:.3f})')
            est_labels.append(y_label)
            est_betas.append(est)
            est_ses.append(est_se)
        plot_est(y=x, labels=est_labels, betas=est_betas, ses=est_ses,
                 xticks=[-0.5, 0, 0.5, 1])
