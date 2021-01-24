import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from skmisc.loess import loess

from maskrcnn.postprocess.analysis import (
    winsorize, load_nightlight_from_point,
    load_building, load_gd_census,
    load_survey, match)


np.random.seed(0)

sns.set(style='ticks', font='Helvetica')
matplotlib.rc('pdf', fonttype=42)

plt.rc('font', size=11)  # controls default text sizes
plt.rc('axes', titlesize=11)  # fontsize of the axes title
plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
plt.rc('legend', fontsize=11)  # legend fontsize
plt.rc('figure', titlesize=11)  # fontsize of the figure title


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
    if scatter:
        ax.plot(x_col, y_col,
                markeredgecolor='none',
                marker='o',
                linestyle='None',
                markersize=3,
                color='dimgrey', alpha=0.07)
    if 'loess' in method:
        m = loess(x_col, y_col, **kwargs)
        m.fit()
        pred = m.predict(x_col, stderror=True).confidence()
        pred_fit = pred.fit
        pred_lower, pred_upper = pred.lower, pred.upper
        # if se:
        #     ax.fill_between(x_col, pred_lower, pred_upper,
        #                     color=color, alpha=.2)
        ax.plot(x_col, pred_fit, '-',
                color=color, linewidth=1.5, alpha=0.7)

    if 'linear' in method:
        X = sm.add_constant(x_col)
        m = sm.OLS(y_col, X).fit()
        pred = m.get_prediction(X)
        pred_fit = pred.predicted_mean
        pred_lower, pred_upper = pred.conf_int().T
        if se:
            ax.fill_between(x_col, pred_lower, pred_upper,
                            color=color, alpha=.1)
        ax.plot(x_col, pred_fit,
                ':' if 'loess' in method else '-',
                color=color, linewidth=1.5, alpha=0.7)


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
                   color=color, se=True, scatter=False,
                   span=0.75)  # span controls smoothing for loess
    else:
        for color_key, df_group in df_nona.groupby(split):
            color = cmap[color_key]
            x_col = df_group[x].values
            y_col = df_group[y].values
            plot_curve(ax=ax, method=method, x_col=x_col, y_col=y_col,
                       color=color, se=True, scatter=False)
        # perform statistical test
        print('+++ F Test +++')
        diff = smf.ols(
            f'{y} ~ {x} * {split}', df_nona).fit()
        f_test = diff.f_test(f'({split} = 0), ({x}:{split} = 0)')
        f_value = f_test.fvalue[0, 0]
        p_value = f_test.pvalue
        print(f'F={f_value:.3f}; p={p_value:.3f}')
        if p_value < 0.01:
            ax.set_title('*F-test: p<0.01')

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        if split is None:
            ax.set_title(y_label, loc='left')
        else:
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


def plot_est(ax, y, labels, betas, ses, y_label,
             ticks=None, lim=None, minor_ticks=None):
    ax.errorbar(
        x=betas, y=range(len(betas)),
        xerr=1.96 * np.array(ses), color='#999999',
        capsize=3, fmt='o', markersize=4)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(labels, ha='left')
    ax.set_ylim(len(betas) - 0.5, -3.5)
    ax.set_xlabel('Treatment Effects (USD PPP)')
    if ticks is not None:
        ax.set_xticks(ticks)
        if minor_ticks is not None:
            ax.set_xticks(minor_ticks, minor=True)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    if lim is not None:
        ax.set_xlim(*lim)
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x', which='both', colors='dimgray')
    ax.tick_params(axis='y', which='both', color='none')
    ax.grid(False)
    ax.set_title(f'b  Estimated v.s. Observed Treatment Effects on {y_label}',
                 loc='left', y=0.8)


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
        'f_assets_housing': 178.47 + 377.14,  # row 6-7
        'f_consumption': 292.98,  # row 1
        'f_housing': 377.14,  # row 7
        'f_assets': 178.47,  # row 6
    }
    obs_se = {
        # row 6-7
        'f_assets_housing': np.sqrt(np.square(24.63) +
                                    np.square(26.37)),
        'f_consumption': 60.09,  # row 1
        'f_housing': 26.37,  # row 7
        'f_assets': 24.63,  # row 6
    }

    # load data
    df_sat = load_building(SAT_IN_DIR, grid=None, agg=False)
    df_svy = load_survey(SVY_IN_DIR)
    df_cen = load_gd_census(
        GPS_FILE=CENSUS_GPS_IN_DIR, MASTER_FILE=CENSUS_MASTER_IN_DIR)
    # match
    df = match(
        df_cen, df_svy, df_sat,
        sat_radius=250 / 111000,
        svy_radius=250 / 111000)  # __ meters / 111000 meters -> degrees
    df.loc[:, 'treat'] = df['treat'].astype(float)
    # load nightlight
    df = load_nightlight_from_point(
        df, NL_IN_DIR,
        lon_col='longitude', lat_col='latitude')
    # eligible sample only
    df = df.loc[df['eligible'] > 0, :]
    # examine data
    print(f'Eligible Sample: {df.shape}')
    # print(df.loc[df['eligible'] > 0.5, :].describe().T)
    # print('Ineligible Sample:')
    # print(df.loc[df['eligible'] < 0.5, :].describe().T)

    # plotting begins
    ys = ['nightlight',
          'area_sum',
          'tin_area_sum']
    y_labels = ['Night Light',
                'Building Footprint',
                'Tin-roof Area']
    y_units = ['(nW·cm-2·sr-1)', '(m2)', '(m2)']
    y_ticks = [[0.3, 0.35, 0.4],
               [200, 300, 400],
               [100, 150, 200, 250]]
    # here, assets refers to non-land, non-housing assets only
    xs = ['f_assets_housing',
          'f_consumption',
          'f_housing',
          'f_assets']
    x_labels = ['Assets Owned',
                'Annual Expenditure',
                'Housing Assets Owned',
                'Non-Housing Assets Owned']
    x_unit = '(USD PPP)'
    x_ticks = [
        [0, 5000, 10000],
        [0, 4000, 8000],
        [0, 3000, 6000],
        [0, 2500, 5000],
    ]
    # winsorize satellite based observations
    for y in ys:
        df.loc[:, y] = winsorize(df[y], 0, 97.5)
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
    for x, x_label, x_tick in zip(
        xs, x_labels, x_ticks,
    ):
        print('-' * 36)
        print(x_label)
        print('Replicate the results from the original paper')
        print('Treatment Effect: {} ({})'.format(
            *reg(df.loc[df['eligible'] > 0, :], x, 'treat')))
        est_labels = ['Observed (Egger et al., 2019)',
                      'Estimated based on ...']
        est_betas = [obs[x], np.nan]
        est_ses = [obs_se[x], np.nan]

        fig = plt.figure(figsize=(6.5, 5))
        gs = fig.add_gridspec(2, 3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])

        fig.suptitle('a  Engel Curves', x=0.07, y=0.95, ha='left')

        for ax, y, y_coef, y_coef_se, y_label, y_unit, y_tick in zip(
            [ax0, ax1, ax2],
            ys, y_coefs, y_coef_ses, y_labels, y_units, y_ticks,
        ):
            # control sample only
            df_control = df.loc[df['treat'] < 1, :]
            plot_engel(
                df=df_control, y=y, x=x, ax=ax,
                method=['linear', 'loess'],
                color=palette[-1],
                y_ticks=y_tick, y_label='\n\n' + y_label + ' ' + y_unit,
                x_ticks=x_tick, x_label='',
            )
            scale, scale_se = reg(df_control, y, x)
            est, est_se = compute_est(y_coef, y_coef_se, scale, scale_se)
            print(f'y: {y}, x: {x}')
            print(f'{est:.3f}, {est_se:.3f} = '
                  f'compute_est({y_coef:.3f}, {y_coef_se:.3f}, '
                  f'{scale:.3f}, {scale_se:.3f})')
            est_labels.append('    ' + y_label)
            est_betas.append(est)
            est_ses.append(est_se)
        fig.text(0.5, 0.45, x_label + ' ' + x_unit, ha='center')
        plot_est(ax=ax3, y=x,
                 labels=est_labels, betas=est_betas, ses=est_ses,
                 y_label=x_label,
                 ticks=[-1000, 0, 1000, 2000],
                 lim=(-3000, 2000),
                 minor_ticks=[-800, -600, -400, -200,
                              200, 400, 600, 800,
                              1200, 1400, 1600, 1800])
        fig.savefig(os.path.join(OUT_DIR, f'{x}-raw.pdf'),
                    bbox_inches='tight', pad_inches=0)

    # test for treatment/control differences
    print('-' * 36)
    print('Test for Treatment/Control Engel Curve Differences')
    fig, axes = plt.subplots(figsize=(6.5, 7), ncols=3, nrows=4)
    for row_idx, (x, x_label, x_tick) in enumerate(zip(
        xs, x_labels, x_ticks,
    )):
        for col_idx, (y, y_label, y_tick) in enumerate(zip(
            ys, y_labels, y_ticks,
        )):
            plot_engel(df=df, y=y, x=x,
                       ax=axes[row_idx, col_idx],
                       split='treat', cmap=cmap,
                       y_ticks=y_tick, y_label=y_label,
                       x_ticks=x_tick, x_label=x_label)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'engel-diff-raw.pdf'),
                bbox_inches='tight', pad_inches=0)
