import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial
from skmisc.loess import loess

from maskrcnn.postprocess.analysis import (
    winsorize, load_nightlight_from_point,
    load_building, load_gd_census)


np.random.seed(0)
matplotlib.rc('pdf', fonttype=42)
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


def plot_curve(ax, method, x_col, y_col, color='dimgrey',
               scatter=True, se=False, **kwargs):
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
    if scatter:
        ax.plot(x_col, y_col,
                markeredgecolor='none',
                marker='o',
                linestyle='None',
                markersize=3,
                color='dimgrey', alpha=0.07)
    if se:
        ax.fill_between(x_col, pred_lower, pred_upper,
                        color=color, alpha=.2)
    ax.plot(x_col, pred_fit,
            color=color, linewidth=1.5, alpha=1)


def plot_engel(df, y, x, ax, split=None,
               method='linear',
               color=None, cmap=None,
               x_label=None, y_label=None,
               x_ticks=None, x_ticklabels=None,
               y_ticks=None, y_ticklabels=None):

    df_nona = df.dropna(subset=[y, x]).sort_values(by=[x])
    if split is None:
        x_col = df_nona[x].values
        y_col = df_nona[y].values
        plot_curve(ax=ax, method=method, x_col=x_col, y_col=y_col,
                   color=color, scatter=True)
    else:
        for color_key, df_group in df_nona.groupby(split):
            color = cmap[color_key]
            x_col = df_group[x].values
            y_col = df_group[y].values
            plot_curve(ax=ax, method=method, x_col=x_col, y_col=y_col,
                       color=color, se=False)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_title(y_label)
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


def plot_est(y, labels, betas, ses, xticks=None):
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.errorbar(
        x=betas, y=range(len(betas)),
        xerr=1.96 * np.array(ses), color='#999999',
        capsize=3, fmt='o')
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, len(betas) - 0.5)
    ax.set_xlabel('Treatment Effects (USD PPP)')
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
    fig.savefig(os.path.join(OUT_DIR, f'betas-{y}.pdf'))


def load_survey(SVY_IN_DIR):
    # load survey data
    df_svy = pd.read_stata(SVY_IN_DIR)
    # print('Observations in raw data: ', df_svy.shape[0])

    # drop households without geo coords
    df_svy = df_svy.dropna(
        subset=['latitude', 'longitude'],
    ).reset_index(drop=True)
    # print('Observations w/ coords: ', df_svy.shape[0])

    # f for final variables
    # calculate per capita consumption / assets
    # convert to USD PPP by dividing by 46.5, per the GiveDirectly paper
    df_svy.loc[:, 'f_consumption'] = winsorize(
        df_svy['p2_consumption_wins'] / 46.5,
        2.5, 97.5)
    df_svy.loc[:, 'f_assets'] = winsorize(
        ((df_svy['p1_assets'] / 46.5) +
         df_svy['h1_11_landvalue_wins_PPP'] +
         df_svy['h1_10_housevalue_wins_PPP']),
        2.5, 97.5)
    df_svy.loc[:, 'f_housing'] = winsorize(
        df_svy['h1_10_housevalue_wins_PPP'],
        2.5, 97.5)

    # check missing
    assert (df_svy.loc[:, ['treat', 'hi_sat', 's1_hhid_key', 'satcluster']]
                  .notna().all().all())

    df_svy.loc[:, 'eligible'] = 1 - df_svy['h1_6_nonthatchedroof_BL']
    # print('Observations in final sample: ', df_svy.shape[0])
    # print('Eligible Sample:')
    # print(df_svy.loc[df_svy['eligible'] > 0.5, :].describe().T)
    # print('Ineligible Sample:')
    # print(df_svy.loc[df_svy['eligible'] < 0.5, :].describe().T)
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
    print(f"Matching {(df_sat['dist'] < radius).sum()} observations")
    print(f"Dropping {(df_sat['dist'] >= radius).sum()} observations")
    df_sat = df_sat.loc[df_sat['dist'] < radius, :]
    # take all the structures within the radius
    df_sat = df_sat.groupby('census_id').agg(
        area_sum=pd.NamedAgg(column='area', aggfunc='sum'),
        tin_area_sum=pd.NamedAgg(column='color_tin_area', aggfunc='sum'),
    ).reset_index()
    # match surveys to households
    dists, cen_idxes = tree.query(
        df_svy.loc[:, ['longitude', 'latitude']].values, k=1)
    df_svy.loc[:, 'dist'] = dists
    df_svy.loc[:, 'census_id'] = cen_idxes
    print(f"Matching {(df_svy['dist'] < radius).sum()} observations")
    print(f"Dropping {(df_svy['dist'] >= radius).sum()} observations")
    df_svy = df_svy.loc[df_svy['dist'] < radius, :]
    df_svy = df_svy.sort_values(by=['census_id', 'dist'])
    df_svy = df_svy.drop_duplicates(subset=['census_id'], keep='first')
    # merge
    df = pd.merge(
        df_cen.loc[:, ['census_id', 'longitude', 'latitude']],
        df_sat.loc[:, ['census_id', 'area_sum', 'tin_area_sum']],
        how='left', on='census_id',
    )
    df.fillna(
        {'house_count': 0, 'area_sum': 0, 'color_tin': 0,
         'color_thatched': 0, 'color_tin_area': 0, 'color_thatched_area': 0},
        inplace=True)
    df = pd.merge(
        df,
        df_svy.loc[:, ['census_id', 's1_hhid_key',
                       'treat', 'eligible', 'hi_sat',
                       'f_consumption', 'f_assets', 'f_housing']],
        how='left', on='census_id',
    )
    # print(df.describe().T)
    return df


if __name__ == '__main__':

    palette = ['#820000', '#ea0000', '#fff4da', '#5d92c4', '#070490']
    cmap = {0: palette[-1], 1: palette[0]}

    SVY_IN_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
    SAT_IN_DIR = 'data/Siaya/Merged/sat.csv'
    NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'
    CENSUS_GPS_IN_DIR = (
        'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv')
    CENSUS_MASTER_IN_DIR = (
        'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
    ATE_IN_DIR = 'output/fig-ate/cache/{}_main.csv'
    OUT_DIR = 'output/fig-engel'

    # 'True' Effect from the original paper
    # https://www.nber.org/system/files/working_papers/w26600/w26600.pdf
    # From Table 1, Column 1
    obs = {
        'f_assets': 178.47 + 377.14 + 49.50,  # row 6-8
        'f_consumption': 292.98,  # row 1
        'f_housing': 377.14,  # row 7
    }
    obs_se = {
        # row 6-8
        'f_assets': np.sqrt(np.square(24.63) +
                            np.square(26.37) +
                            np.square(186.30)),
        'f_consumption': 60.09,  # row 1
        'f_housing': 26.37,  # row 7
    }

    # load data
    df_sat = load_building(SAT_IN_DIR, grid=None, agg=False)
    df_svy = load_survey(SVY_IN_DIR)
    df_cen = load_gd_census(
        GPS_FILE=CENSUS_GPS_IN_DIR, MASTER_FILE=CENSUS_MASTER_IN_DIR)

    # match
    df = match(df_cen, df_svy, df_sat,
               radius=0.0009)  # = 100m
    df.loc[:, 'treat'] = df['treat'].astype(float)

    # load nightlight
    df = load_nightlight_from_point(
        df, NL_IN_DIR,
        lon_col='longitude', lat_col='latitude')
    # eligible sample only
    df = df.loc[df['eligible'] > 0, :]
    # examine data
    # print('Eligible Sample:')
    # print(df.loc[df['eligible'] > 0.5, :].describe().T)
    # print('Ineligible Sample:')
    # print(df.loc[df['eligible'] < 0.5, :].describe().T)

    # plotting begins
    ys = ['nightlight',
          'area_sum',
          'tin_area_sum']
    y_labels = ['Night Light',
                'Building Footprint (sq meters)',
                'Tin-roof Area (sq meters)']
    y_ticks = [None, None, None]
    y_ticklabels = [None, None, None]
    xs = ['f_assets',
          'f_consumption',
          'f_housing']
    x_labels = ['Assets (USD PPP)',
                'Consumption (USD PPP)',
                'Housing Asset (USD PPP)']
    x_ticks = [None, None, None]
    x_ticklabels = [None, None, None]
    # winsorize satellite based observations
    for y in ys:
        df.loc[:, y] = winsorize(df[y], 2.5, 97.5)
    # load previous estimates
    y_coefs = []
    y_coef_ses = []
    for y in ys:
        df_y = pd.read_csv(ATE_IN_DIR.format(y))
        y_coef, = df_y.loc[df_y['x'].isna(), 'beta']
        y_coef_se, = df_y.loc[df_y['x'].isna(), 'se']
        y_coefs.append(y_coef)
        y_coef_ses.append(y_coef_se)

    # double loop
    for x, x_label, x_tick, x_ticklabel in zip(
        xs, x_labels, x_ticks, x_ticklabels
    ):
        print('-' * 36)
        print(x_label)
        print('Replicate the results from the original paper')
        print('Treatment Effect: {} ({})'.format(
            *reg(df.loc[df['eligible'] > 0, :], x, 'treat')))
        est_labels = ['Observed (Egger et al., 2019)']
        est_betas = [obs[x]]
        est_ses = [obs_se[x]]

        fig, axes = plt.subplots(figsize=(9, 3), ncols=3)
        for ax, y, y_coef, y_coef_se, y_label, y_tick, y_ticklabel in zip(
            axes, ys, y_coefs, y_coef_ses, y_labels, y_ticks, y_ticklabels
        ):
            plot_engel(
                df=df,
                y=y,
                x=x,
                ax=ax,
                color=palette[0],
                y_ticks=y_tick,
                y_ticklabels=y_ticklabel,
                y_label=y_label,
                x_ticks=x_tick,
                x_ticklabels=x_ticklabel,
                x_label=x_label,
            )
            scale, scale_se = reg(df, y, x)
            est, est_se = compute_est(y_coef, y_coef_se, scale, scale_se)
            print(f'y: {y}, x: {x}')
            print(f'{est:.3f}, {est_se:.3f} = '
                  f'compute_est({y_coef:.3f}, {y_coef_se:.3f}, '
                  f'{scale:.3f}, {scale_se:.3f})')
            est_labels.append(y_label)
            est_betas.append(est)
            est_ses.append(est_se)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'engel-{x}.pdf'))
        plot_est(y=x, labels=est_labels, betas=est_betas, ses=est_ses,
                 xticks=None)
