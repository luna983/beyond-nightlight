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
    load_building)


np.random.seed(0)
sns.set(style='ticks', font='Helvetica', font_scale=1)


def plot(df, y, x,
         x_ticks, x_ticklabels, y_ticks_l, y_ticklabels_l,
         y_ticks_r,
         treat='treat',
         cmap={0: '#2c7bb6', 1: '#d7191c'},
         method='loess',
         x_label='', y_label_l='', y_label_r=''):
    loess_params = {'degree': 1}
    df_nona = df.dropna(subset=[x, y]).sort_values(by=x)
    # regression
    results = smf.ols(y + ' ~ ' + treat, data=df_nona).fit()
    y_coef = results.params[treat]
    y_se = results.bse[treat]
    # y_pvalue = results.pvalues[treat]
    results = smf.ols(x + ' ~ ' + treat, data=df_nona).fit()
    x_coef = results.params[treat]
    x_se = results.bse[treat]
    # x_pvalue = results.pvalues[treat]
    results = sm.OLS(df_nona[y].values,
                     sm.add_constant(df_nona[x].values)).fit()
    scale = results.params[1]
    scale_se = results.bse[1]
    # calculate estimated effect
    est = y_coef / scale
    est_se = np.sqrt((y_se / y_coef) ** 2 + (scale_se / scale) ** 2) * abs(est)
    # make figure
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 4))
    for cmap_value, cmap_color in cmap.items():
        y_col = df_nona.loc[df_nona[treat] == cmap_value, y]
        x_col = df_nona.loc[df_nona[treat] == cmap_value, x]
        if method == 'loess':
            m = loess(x_col, y_col, **loess_params)
            m.fit()
            pred = m.predict(x_col, stderror=True).confidence()
            pred_fit = pred.fit
            pred_lower, pred_upper = pred.lower, pred.upper
        elif method == 'linear':
            results = sm.OLS(y_col,
                             sm.add_constant(x_col)).fit()
            pred = results.get_prediction(sm.add_constant(x_col))
            pred_fit = pred.predicted_mean
            pred_lower, pred_upper = pred.conf_int().T
        else:
            raise NotImplementedError
        ax0.plot(x_col, pred_fit, color=cmap_color, linewidth=2, alpha=0.8)
        # ax0.fill_between(x_col, pred_lower, pred_upper,
        #                  color=cmap_color, alpha=.2)
    if method == 'loess':
        m = loess(df_nona[x].values, df_nona[y].values, **loess_params)
        m.fit()
        pred = m.predict(df_nona[x].values, stderror=True).confidence()
        pred_fit = pred.fit
        pred_lower, pred_upper = pred.lower, pred.upper
    elif method == 'linear':
        results = sm.OLS(df_nona[y].values,
                         sm.add_constant(df_nona[x].values)).fit()
        pred = results.get_prediction(sm.add_constant(df_nona[x].values))
        pred_fit = pred.predicted_mean
        pred_lower, pred_upper = pred.conf_int().T
    else:
        raise NotImplementedError
    ax0.plot(df_nona[x].values, pred_fit,
             color='dimgray', linewidth=1, alpha=0.4)
    ax0.fill_between(df_nona[x].values, pred_lower, pred_upper,
                     color='dimgray', alpha=.2)
    ax0.set_title(
        f'N = {df_nona.shape[0]}\n'
        f'Observed effects: {x_coef:.4f}\n'
        f'95% CI: [{x_coef - 1.96 * x_se:.4f}, {x_coef + 1.96 * x_se:.4f}]\n'
        f'Estimated effects: {y_coef:.4f} / {scale:.4f} = {est:.4f}\n'
        f'95% CI: [{est - 1.96 * est_se:.4f}, {est + 1.96 * est_se:.4f}]\n')
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label_l)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_ticklabels)
    ax0.set_yticks(y_ticks_l)
    ax0.set_yticklabels(y_ticklabels_l)
    ax0.spines['left'].set_bounds(ax0.get_yticks()[0], ax0.get_yticks()[-1])
    ax0.spines['left'].set_color('dimgray')
    ax0.spines['bottom'].set_bounds(ax0.get_xticks()[0], ax0.get_xticks()[-1])
    ax0.spines['bottom'].set_color('dimgray')
    ax0.spines['right'].set_color('none')
    ax0.spines['top'].set_color('none')
    ax0.tick_params(axis='x', colors='dimgray')
    ax0.tick_params(axis='y', colors='dimgray')
    ax0.grid(False)
    ax1.errorbar(0, est, yerr=1.96 * est_se, color='#d7191c',
                 capsize=3, fmt='--o')
    ax1.errorbar(1, x_coef, yerr=1.96 * x_se, color='#999999',
                 capsize=3, fmt='--o')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Estimated', 'Observed'])
    ax1.set_yticks(y_ticks_r)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(y_ticks_r[0], y_ticks_r[-1])
    ax1.set_ylabel(y_label_r)
    ax1.spines['left'].set_bounds(ax1.get_yticks()[0], ax1.get_yticks()[-1])
    ax1.spines['left'].set_color('dimgray')
    ax1.spines['bottom'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.tick_params(axis='y', colors='dimgray')
    ax1.tick_params(axis='x', color='none')
    ax1.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{y}-{x}.pdf'))


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
    df_svy = (df_svy.loc[df_svy['h1_6_nonthatchedroof_BL'] == 0, :]
                    .reset_index(drop=True).copy())
    print('Observations in final sample: ', df_svy.shape[0])

    return df_svy


def match(
    df_svy, df_sat,
    radius=0.0008,
    # radius=0.00045,  # = 50m
    k=20,  # no. of nearest neighbors examined
):
    # match structures to households
    # one structure is matched to one household at most
    tree = scipy.spatial.cKDTree(
        df_svy.loc[:, ['longitude', 'latitude']].values)
    dists, svy_idxes = tree.query(
        df_sat.loc[:, ['centroid_lon', 'centroid_lat']].values, k=k)
    rank, sat_idxes = np.meshgrid(range(k), range(df_sat.shape[0]))
    assert (dists[:, -1] > radius).all(), 'increase k value'

    svy_idxes = svy_idxes[dists < radius]
    sat_idxes = sat_idxes[dists < radius]
    df = pd.concat([
        df_svy.loc[svy_idxes, ['s1_hhid_key']].reset_index(drop=True),
        df_sat.loc[sat_idxes, :].reset_index(drop=True),
        pd.DataFrame({'rank': rank[dists < radius],
                      'distance': dists[dists < radius]}),
    ], axis=1)
    df = df.sort_values(by=['s1_hhid_key', 'distance'])

    # option A: take the closest structure
    df_close = df.drop_duplicates(subset=['s1_hhid_key'], keep='first')
    df_close = pd.merge(df_svy, df_close, how='left', on=['s1_hhid_key'])
    df_close.loc[:, 'area_pc'] = (df_close['area'].values /
                                  df_close['hhsize1_BL'].values)

    # option B: take all the structures within the radius
    df_circle = df.groupby('s1_hhid_key').agg(
        house_count=pd.NamedAgg(column='area', aggfunc='count'),
        area_sum=pd.NamedAgg(column='area', aggfunc='sum'),
        # RGB_mean=pd.NamedAgg(column='RGB_mean', aggfunc='mean'),
        # RGB_mean_spline=pd.NamedAgg(
        #     column='RGB_mean_spline', aggfunc='mean'),
        color_tin=pd.NamedAgg(column='color_tin', aggfunc='sum'),
        color_thatched=pd.NamedAgg(column='color_thatched', aggfunc='sum'),
        color_tin_area=pd.NamedAgg(column='color_tin_area', aggfunc='sum'),
        color_thatched_area=pd.NamedAgg(
            column='color_thatched_area', aggfunc='sum'),
    ).reset_index()
    df_circle = pd.merge(df_svy, df_circle, how='left', on=['s1_hhid_key'])
    df_circle.fillna(
        {'house_count': 0, 'area_sum': 0, 'color_tin': 0,
         'color_thatched': 0, 'color_tin_area': 0, 'color_thatched_area': 0},
        inplace=True)
    df_circle.loc[:, 'area_sum_pc'] = (df_circle['area_sum'].values /
                                       df_circle['hhsize1_BL'].values)
    df_circle.loc[:, 'color_tin_area_pc'] = (
        df_circle['color_tin_area'].values /
        df_circle['hhsize1_BL'].values)
    df_circle.loc[:, 'log1_area_sum_pc'] = df_circle['area_sum_pc'].apply(
        lambda x: np.log(x + 1) if x > 0 else np.nan
    )
    df_circle.loc[:, 'log1_color_tin_area_pc'] = (
        df_circle['color_tin_area_pc'].apply(
            lambda x: np.log(x + 1) if x > 0 else np.nan
        )
    )
    return df_close, df_circle


if __name__ == '__main__':

    palette = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']

    SVY_IN_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
    SAT_IN_DIR = 'data/Siaya/Merged/sat.csv'
    NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'

    OUT_DIR = 'output/fig-engel'

    # load data
    df_sat = load_building(SAT_IN_DIR, grid=None, agg=False)
    df_svy = load_survey(SVY_IN_DIR)

    # match
    df_close, df_circle = match(df_svy, df_sat)

    # load nightlight
    df_circle = load_nightlight_from_point(
        df_circle, NL_IN_DIR,
        lon_col='longitude', lat_col='latitude')

    # plotting begins
    plot(
        df=df_circle,
        y='sat_nightlight_winsnorm',
        y_ticks_l=[-1, 0, 1],
        y_ticklabels_l=[-1, 0, 1],
        y_label_l='Normalized Nightlight Values',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_assets_all_pc',
        x_ticks=np.log([50, 100, 300, 1000, 3000]),
        x_ticklabels=[50, 100, 300, 1000, 3000],
        x_label='Assets per capita (USD PPP)',
        y_label_r='Effects on log(Assets per capita)',
    )

    plot(
        df=df_circle,
        y='sat_nightlight_winsnorm',
        y_ticks_l=[-1, 0, 1],
        y_ticklabels_l=[-1, 0, 1],
        y_label_l='Normalized Nightlight Values',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_p2_consumption_wins_pc',
        x_ticks=np.log([100, 300, 1000, 3000]),
        x_ticklabels=[100, 300, 1000, 3000],
        x_label='Consumption per capita (USD PPP)',
        y_label_r='Effects on log(Consumption per capita)',
    )

    plot(
        df=df_circle,
        y='area_sum_pc',
        y_ticks_l=[50, 100, 150],
        y_ticklabels_l=[50, 100, 150],
        y_label_l='Building footprint per capita (sq meters)',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_assets_all_pc',
        x_ticks=np.log([50, 100, 300, 1000, 3000]),
        x_ticklabels=[50, 100, 300, 1000, 3000],
        x_label='Assets per capita (USD PPP)',
        y_label_r='Effects on log(Assets per capita)',
    )

    plot(
        df=df_circle,
        y='area_sum_pc',
        y_ticks_l=[50, 100, 150],
        y_ticklabels_l=[50, 100, 150],
        y_label_l='Building footprint per capita (sq meters)',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_p2_consumption_wins_pc',
        x_ticks=np.log([100, 300, 1000, 3000]),
        x_ticklabels=[100, 300, 1000, 3000],
        x_label='Consumption per capita (USD PPP)',
        y_label_r='Effects on log(Consumption per capita)',
    )

    plot(
        df=df_circle,
        y='color_tin_area_pc',
        y_ticks_l=[0, 50, 100],
        y_ticklabels_l=[0, 50, 100],
        y_label_l='Tin-roof area per capita (sq meters)',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_assets_all_pc',
        x_ticks=np.log([50, 100, 300, 1000, 3000]),
        x_ticklabels=[50, 100, 300, 1000, 3000],
        x_label='Assets per capita (USD PPP)',
        y_label_r='Effects on log(Assets per capita)',
    )

    plot(
        df=df_circle,
        y='color_tin_area_pc',
        y_ticks_l=[0, 50, 100],
        y_ticklabels_l=[0, 50, 100],
        y_label_l='Tin-roof area per capita (sq meters)',
        y_ticks_r=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
        x='logwins_p2_consumption_wins_pc',
        x_ticks=np.log([100, 300, 1000, 3000]),
        x_ticklabels=[100, 300, 1000, 3000],
        x_label='Consumption per capita (USD PPP)',
        y_label_r='Effects on log(Consumption per capita)',
    )
