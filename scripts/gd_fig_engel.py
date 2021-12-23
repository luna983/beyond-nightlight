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


def make_broken_axis(ax,
                     x_label=None, y_label=None, y_label_as_title=True,
                     x_ticks=None, x_ticklabels=None,
                     y_ticks=None, y_ticklabels=None):
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        if y_label_as_title:
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


def plot_engel(df, y, x, ax,
               method='linear', scatter=False,
               loess_se=False, linear_se=True,
               color='dimgrey',
               # span controls smoothing for loess
               span=0.75, verbose=False):

    df_nona = df.dropna(subset=[y, x]).sort_values(by=[x])
    x_col = df_nona[x].values
    y_col = df_nona[y].values
    if verbose:
        print(f'Engel curve: N = {df_nona.shape[0]}')
    if scatter:
        ax.plot(x_col, y_col,
                markeredgecolor='none',
                marker='o',
                linestyle='None',
                markersize=3,
                color='dimgrey', alpha=0.07)
    if 'loess' in method:
        m = loess(x_col, y_col, span=span)
        m.fit()
        pred = m.predict(x_col, stderror=True).confidence()
        pred_fit = pred.fit
        pred_lower, pred_upper = pred.lower, pred.upper
        if loess_se:
            ax.fill_between(x_col, pred_lower, pred_upper,
                            color=color, alpha=.2)
        ax.plot(x_col, pred_fit, '-',
                color=color, linewidth=1.5, alpha=0.7)
        _ = test_linearity(x_col, y_col, n_knots=5)

    if 'linear' in method:
        X = sm.add_constant(x_col)
        m = sm.OLS(y_col, X).fit()
        pred = m.get_prediction(X)
        pred_fit = pred.predicted_mean
        pred_lower, pred_upper = pred.conf_int().T
        # standard error
        if linear_se:
            ax.fill_between(x_col, pred_lower, pred_upper,
                            color=color, alpha=.1)
        ax.plot(x_col, pred_fit,
                ':' if 'loess' in method else '-',
                color=color, linewidth=1.5, alpha=0.7)


def plot_est(ax, labels, betas, ses, est_colors, y_label,
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
        if beta < xmin:
            ax.arrow(x=xmin + 1, y=y_pos, dx=-1, dy=0,
                     color=color, width=0.05, head_width=0.4,
                     head_length=100)
        else:
            lower = max(beta - 1.96 * se, xmin) - beta
            lower_clipped = xmin > (beta - 1.96 * se)
            ax.arrow(x=beta, y=y_pos, dx=lower, dy=0,
                     color=color, width=0.05, head_width=0.4,
                     head_length=(100 if lower_clipped else 1))
        if beta > xmax:
            ax.arrow(x=xmax - 1, y=y_pos, dx=1, dy=0,
                     color=color, width=0.05, head_width=0.4,
                     head_length=100)
        else:
            upper = min(beta + 1.96 * se, xmax) - beta
            upper_clipped = xmax < (beta + 1.96 * se)
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
    ax.set_title(r'b    Treatment Effect Estimates on ' + y_label,
                 loc='left', x=-0.09, y=0.65)


def collate_data(SAT_IN_DIR, SVY_IN_DIR,
                 CENSUS_GPS_IN_DIR, CENSUS_MASTER_IN_DIR,
                 NL_IN_DIR):
    rename_vars = {
        'area_sum': 'sat_building_footprint',
        'tin_area_sum': 'sat_tin_roof_area',
        'nightlight': 'sat_night_light',
        'eligible': 'is_eligible',
        'treat': 'in_treatment_group'}
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
    # rename variables
    df = df.rename(rename_vars, axis=1)

    # subset samples
    print('[Matched] Matched sample: treat = ',
          df.loc[(df['is_eligible'] == 1) &
                 (df['in_treatment_group'] == 1), :].shape[0],
          '; control = ',
          df.loc[(df['is_eligible'] == 1) &
                 (df['in_treatment_group'] == 0), :].shape[0],
          '; ineligible = ',
          df.loc[df['is_eligible'] == 0, :].shape[0])

    df = df.loc[df['svy_housing_assets'] > 0, :]
    print('[Matched] Matched sample, dropping renters: treat = ',
          df.loc[(df['is_eligible'] == 1) &
                 (df['in_treatment_group'] == 1), :].shape[0],
          '; control = ',
          df.loc[(df['is_eligible'] == 1) &
                 (df['in_treatment_group'] == 0), :].shape[0],
          '; ineligible = ',
          df.loc[df['is_eligible'] == 0, :].shape[0])

    df_eligible = df.loc[df['is_eligible'] == 1, :].copy()
    df_ineligible = df.loc[df['is_eligible'] == 0, :].copy()

    wins_lower_bound = 1
    wins_upper_bound = 99
    # winsorize survey based observations
    for column in [col for col in df.columns if col.startswith('svy_')]:
        df_eligible.loc[:, column] = winsorize(
            df_eligible[column], wins_lower_bound, wins_upper_bound)
        df_ineligible.loc[:, column] = winsorize(
            df_ineligible[column], wins_lower_bound, wins_upper_bound)
    # winsorize satellite based observations
    for column in [col for col in df.columns if col.startswith('sat_')]:
        df_eligible.loc[:, column] = winsorize(
            df_eligible[column], 0, wins_upper_bound)
        df_ineligible.loc[:, column] = winsorize(
            df_ineligible[column], 0, wins_upper_bound)
    return df_ineligible, df_eligible


if __name__ == '__main__':

    SVY_IN_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
    SAT_IN_DIR = 'data/Siaya/Merged/sat.csv'
    NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'
    CENSUS_GPS_IN_DIR = (
        'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv')
    CENSUS_MASTER_IN_DIR = (
        'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')
    ATE_IN_DIR = 'fig_raw_data/fig-ate.csv'
    FIG_OUT_DIR = 'output/fig-engel'
    RAW_DATA_OUT_DIR = 'fig_raw_data'

    # 'True' Effect from the original paper
    # https://www.nber.org/system/files/working_papers/w26600/w26600.pdf
    # From Table 1, Column 1
    obs = {
        'total_assets': 178.47 + 377.14,  # row 6-7
        'annual_expenditure': 292.98,  # row 1
        'housing_assets': 377.14,  # row 7
        'non_housing_assets': 178.47,  # row 6
    }
    obs_se = {
        # row 6-7
        'total_assets': np.sqrt(np.square(24.63) +
                                np.square(26.37)),
        'annual_expenditure': 60.09,  # row 1
        'housing_assets': 26.37,  # row 7
        'non_housing_assets': 24.63,  # row 6
    }

    cmap = {'control': '#666666', 'treat': '#0F9D58'}
    y_colors = ['#DB4437', '#4285F4', '#F4B400']
    ys = ['building_footprint',
          'tin_roof_area',
          'night_light']
    y_labels = ['Building Footprint',
                'Tin-roof Area',
                'Night Light']
    y_units = [r'$\mathregular{(m^2)}$',
               r'$\mathregular{(m^2)}$',
               r'$\mathregular{(nW·cm^{-2}·sr^{-1})}$']
    y_ticks = [[200, 250, 300],
               [100, 150, 200],
               [0.3, 0.33, 0.36]]
    # here, assets refers to non-land, non-housing assets only
    xs = ['total_assets',
          'annual_expenditure',
          'housing_assets',
          'non_housing_assets']
    x_labels = ['Total Assets',
                'Annual Expenditure',
                'Housing Assets',
                'Non-Housing Assets']
    x_unit = '(USD PPP)'
    x_ticks = [
        [0, 3000, 6000],
        [0, 4500, 9000],
        [0, 2000, 4000],
        [0, 2000, 4000],
    ]

    # load previous estimates
    df_y = pd.read_csv(ATE_IN_DIR)
    df_y = df_y.loc[df_y['x'].isna(), ['beta', 'se', 'outcome']].copy()
    df_y = df_y.set_index('outcome', drop=True)

    df_ineligible, df_eligible = collate_data(
        SAT_IN_DIR, SVY_IN_DIR,
        CENSUS_GPS_IN_DIR, CENSUS_MASTER_IN_DIR,
        NL_IN_DIR)

    df_control = df_eligible.loc[df_eligible['in_treatment_group'] == 0, :]
    df_treat = df_eligible.loc[df_eligible['in_treatment_group'] == 1, :]

    # ================================================================
    # output raw data
    output_cols = [col for col in df_control.columns
                   if col.startswith(('sat_', 'svy_'))]
    df_control.loc[:, output_cols].to_csv(
        os.path.join(RAW_DATA_OUT_DIR, 'fig-engel-a.csv'), index=False)
    est_output_raw_data = []

    # double loop
    for x, x_label, x_tick in zip(
        xs, x_labels, x_ticks,
    ):
        print('-' * 72)
        print(x_label)
        print('In-sample Treatment Effect Estimate: {:.1f} ({:.1f})'.format(
              *reg(df_eligible, 'svy_' + x, 'in_treatment_group')))
        est_labels = ['Survey-based estimate',
                      'Satellite-derived estimates based on: ']
        est_betas = [obs[x], np.nan]
        est_ses = [obs_se[x], np.nan]
        est_output_raw_data.append({
            'outcome': x, 'type': est_labels[0],
            'beta': obs[x], 'standard_error': obs_se[x],
        })

        fig = plt.figure(figsize=(6, 5))
        plt.subplots_adjust(wspace=0.35)
        gs = fig.add_gridspec(2, 3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, :])

        fig.suptitle(r'a    Engel Curves', x=0.05, y=0.97, ha='left')

        for ax, y, y_label, y_unit, y_tick, y_color in zip(
            [ax0, ax1, ax2],
            ys, y_labels, y_units, y_ticks, y_colors,
        ):
            print(f'---- Variable: {y_label} ----')
            print('Engel Curve Statistics: ')
            # control sample only
            plot_engel(
                df=df_control, y='sat_' + y, x='svy_' + x, ax=ax,
                method=['linear', 'loess'], scatter=False,
                color=y_color)
            make_broken_axis(
                ax=ax,
                y_ticks=y_tick, y_label='\n\n' + y_label + ' ' + y_unit,
                x_ticks=x_tick,
            )
            y_coef, y_coef_se = df_y.loc[y, ['beta', 'se']].values
            scale, scale_se = reg(
                df_control, 'sat_' + y, 'svy_' + x, verbose=True)
            est, est_se = compute_est(y_coef, y_coef_se, scale, scale_se)
            print(f'Remotely-sensed Effects: {y_coef:.3f} ({y_coef_se:.3f})\n'
                  f'Scaling Factor: {scale:.6f} ({scale_se:.6f})\n'
                  f'Scaled Effect: {est:.3f} ({est_se:.3f}) USD PPP')
            est_labels.append('    ' + y_label)
            est_betas.append(est)
            est_ses.append(est_se)
            est_output_raw_data.append({
                'outcome': x, 'type': est_labels[1] + y_label,
                'beta': est, 'standard_error': est_se,
            })
        fig.text(0.5, 0.43, x_label + ' ' + x_unit, ha='center')
        plot_est(ax=ax3,
                 labels=est_labels, betas=est_betas, ses=est_ses,
                 est_colors=['dimgray'] * 2 + y_colors, y_label=x_label,
                 ticks=[-1000, 0, 1000, 2000],
                 lim=(-2500, 2100),
                 minor_ticks=[-800, -600, -400, -200,
                              200, 400, 600, 800,
                              1200, 1400, 1600, 1800])
        fig.savefig(os.path.join(
            FIG_OUT_DIR, f"fig-engel-{x.replace('_', '-')}-raw.pdf"))

    # output raw data for panel b
    pd.DataFrame(est_output_raw_data).to_csv(os.path.join(
        RAW_DATA_OUT_DIR, 'fig-engel-b.csv'), index=False)

    # ================================================================
    x = xs[0]
    x_label = x_labels[0]
    x_tick = x_ticks[0]
    # subset ineligible sample
    df_ineligible = df_ineligible.loc[
        df_ineligible['svy_' + x] < df_eligible['svy_' + x].max(), :].copy()
    # output raw data
    output_cols = ([col for col in df_control.columns
                    if col.startswith(('sat_'))] +
                   ['svy_' + x, 'is_eligible', 'in_treatment_group'])
    output_raw_data = pd.concat(
        [df_eligible.loc[:, output_cols],
         df_ineligible.loc[:, output_cols]])
    output_raw_data.loc[:, 'is_eligible'] = (
        output_raw_data['is_eligible'].astype(int))
    output_raw_data.loc[:, 'in_treatment_group'] = (
        output_raw_data['in_treatment_group'].astype(int))
    output_raw_data.to_csv(
        os.path.join(RAW_DATA_OUT_DIR, 'fig-engel-diff.csv'), index=False)

    # test for treatment/control differences
    print('-' * 72)
    print('Diff: Treatment versus Control')
    fig, axes = plt.subplots(figsize=(6.5, 2.5), ncols=3)
    for col_idx, (y, y_label, y_tick) in enumerate(zip(
        ys, y_labels, y_ticks,
    )):
        print(f'---- Variable: {y_label} ----')
        ax = axes[col_idx]
        shared_args = dict(method=['linear'],
                           y='sat_' + y, x='svy_' + x, ax=ax, verbose=True)
        print('[Treatmen group]')
        plot_engel(df=df_treat, color=cmap['treat'], **shared_args)
        print('[Control group]')
        plot_engel(df=df_control, color=cmap['control'], **shared_args)
        make_broken_axis(
            ax=ax,
            y_ticks=y_tick, y_label=y_label, y_label_as_title=True,
            x_ticks=x_tick)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    fig.text(0.5, 0.02, x_label + ' ' + x_unit, ha='center')
    fig.savefig(os.path.join(FIG_OUT_DIR, f'fig-engel-tc-diff-raw.pdf'))

    # ================================================================
    # test for eligible/ineligible differences
    print('-' * 72)
    print('Diff: Eligible vs Ineligible')
    fig, axes = plt.subplots(figsize=(6.5, 2.5), ncols=3)
    for col_idx, (y, y_label, y_tick) in enumerate(zip(
        ys, y_labels, y_ticks,
    )):
        print(f'---- Variable: {y_label} ----')
        ax = axes[col_idx]
        shared_args = dict(method=['linear'],
                           y='sat_' + y, x='svy_' + x, ax=ax, verbose=True)
        print('[Eligible sample]')
        plot_engel(df=df_eligible, color=cmap['control'], **shared_args)
        print('[Ineligible sample]')
        plot_engel(df=df_ineligible, color=cmap['treat'], **shared_args)
        make_broken_axis(
            ax=ax,
            y_ticks=y_tick, y_label=y_label, y_label_as_title=True,
            x_ticks=x_tick)
    fig.subplots_adjust(wspace=0.3, bottom=0.2)
    fig.text(0.5, 0.02, x_label + ' ' + x_unit, ha='center')
    fig.savefig(os.path.join(FIG_OUT_DIR, f'fig-engel-ei-diff-raw.pdf'))
