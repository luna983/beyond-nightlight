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
    load_survey, match, test_linearity)


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


def reg(df, y, x, verbose=False):
    df_nona = df.dropna(subset=[x, y])
    results = smf.ols(y + ' ~ ' + x, data=df_nona).fit()
    beta = results.params[x]
    se = results.bse[x]
    if verbose:
        print(f'N = {df_nona.shape[0]}, R2 = {results.rsquared:.4f}')
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
        p_value = test_linearity(x_col, y_col, n_knots=5)
        print(f'Test for Linearity: p = {p_value:.3f}')

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


def plot_est(ax, y, labels, betas, ses, est_colors, y_label,
             ticks=None, lim=None, minor_ticks=None):
    if ticks is not None:
        ax.set_xticks(ticks)
        if minor_ticks is not None:
            ax.set_xticks(minor_ticks, minor=True)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])
    if lim is not None:
        ax.set_xlim(*lim)
    xmin, xmax = ticks[0], ticks[-1]
    for y_pos, (beta, se, color) in enumerate(zip(betas, ses, est_colors)):
        lower = max(beta - 1.96 * se, xmin) - beta
        upper = min(beta + 1.96 * se, xmax) - beta
        lower_clipped = xmin > (beta - 1.96 * se)
        upper_clipped = xmax < (beta + 1.96 * se)
        ax.arrow(x=beta, y=y_pos, dx=lower, dy=0,
                 color=color, width=0.05, head_width=0.4,
                 head_length=(100 if lower_clipped else 1))
        ax.arrow(x=beta, y=y_pos, dx=upper, dy=0,
                 color=color, width=0.05, head_width=0.4,
                 head_length=(100 if upper_clipped else 1))
        ax.plot(
            beta, y_pos, marker='o', color=color, markersize=5)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(labels, ha='left')
    ax.set_ylim(len(betas) - 0.5, -3.5)
    ax.set_xlabel('Treatment Effect (USD PPP)')
    ax.xaxis.set_label_coords(0.65, -0.2)

    ax.spines['bottom'].set_color('dimgray')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x', which='both', colors='dimgray')
    ax.tick_params(axis='y', which='both', color='none')
    ax.grid(False)
    ax.set_title(r'$\bf{b}$    Treatment Effect Estimates on ' + y_label,
                 loc='left', x=-0.09, y=0.65)


if __name__ == '__main__':

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
    print('[Matched] Matched sample: N =', df.shape[0])

    # plotting begins
    cmap = {0: '#666666', 1: '#0F9D58'}
    y_colors = ['#DB4437', '#4285F4', '#F4B400']
    ys = ['area_sum',
          'tin_area_sum',
          'nightlight']
    y_labels = ['Building Footprint',
                'Tin-roof Area',
                'Night Light']
    y_units = [r'$\mathregular{(m^2)}$',
               r'$\mathregular{(m^2)}$',
               r'$\mathregular{(nW·cm^{-2}·sr^{-1})}$']
    y_ticks = [[200, 300, 400],
               [100, 150, 200, 250],
               [0.3, 0.35, 0.4]]
    # here, assets refers to non-land, non-housing assets only
    xs = ['f_assets_housing',
          'f_consumption',
          'f_housing',
          'f_assets']
    x_labels = ['Total Assets',
                'Annual Expenditure',
                'Housing Assets',
                'Non-Housing Assets']
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
        print('-' * 72)
        print(x_label)
        print('In-sample Treatment Effect Estimate: {:.1f} ({:.1f})'.format(
              *reg(df.loc[df['eligible'] > 0, :], x, 'treat')))
        est_labels = ['Survey-based estimate',
                      'Satellite-derived estimates based on ...']
        est_betas = [obs[x], np.nan]
        est_ses = [obs_se[x], np.nan]

        fig = plt.figure(figsize=(6, 5))
        plt.subplots_adjust(wspace=0.35)
        gs = fig.add_gridspec(2, 3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])

        fig.suptitle(r'$\bf{a}$    Engel Curves', x=0.05, y=0.97, ha='left')

        for ax, y, y_coef, y_coef_se, y_label, y_unit, y_tick, y_color in zip(
            [ax0, ax1, ax2],
            ys, y_coefs, y_coef_ses, y_labels, y_units, y_ticks, y_colors,
        ):
            print(f'---- Variable: {y_label} ----')
            print('Engel Curve Statistics: ')
            # control sample only
            df_control = df.loc[df['treat'] < 1, :]
            plot_engel(
                df=df_control, y=y, x=x, ax=ax,
                method=['linear', 'loess'],
                color=y_color,
                y_ticks=y_tick, y_label='\n\n' + y_label + ' ' + y_unit,
                x_ticks=x_tick, x_label='',
            )
            scale, scale_se = reg(df_control, y, x, verbose=True)
            est, est_se = compute_est(y_coef, y_coef_se, scale, scale_se)
            print(f'Remotely-sensed Effects: {y_coef:.3f} ({y_coef_se:.3f})\n'
                  f'Scaling Factor: {scale:.6f} ({scale_se:.6f})\n'
                  f'Scaled Effect: {est:.3f} ({est_se:.3f}) USD PPP')
            est_labels.append('    ' + y_label)
            est_betas.append(est)
            est_ses.append(est_se)
        fig.text(0.5, 0.43, x_label + ' ' + x_unit, ha='center')
        plot_est(ax=ax3, y=x,
                 labels=est_labels, betas=est_betas, ses=est_ses,
                 est_colors=['dimgray'] * 2 + y_colors, y_label=x_label,
                 ticks=[-1000, 0, 1000, 2000],
                 lim=(-2500, 2100),
                 minor_ticks=[-800, -600, -400, -200,
                              200, 400, 600, 800,
                              1200, 1400, 1600, 1800])
        fig.savefig(os.path.join(OUT_DIR, f'{x}-raw.pdf'))

    # test for treatment/control differences
    fig, axes = plt.subplots(figsize=(6, 6.5), ncols=3, nrows=4)
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
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    fig.savefig(os.path.join(OUT_DIR, f'engel-diff-raw.pdf'))
