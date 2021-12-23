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
from gd_fig_engel import collate_data


SVY_IN_DIR = 'data/External/GiveDirectly/GE_Luna_Extract_2020-07-27.dta'
SAT_IN_DIR = 'data/Siaya/Merged/sat.csv'
NL_IN_DIR = 'data/External/Nightlight/VIIRS_DNB_KE_2019.tif'
CENSUS_GPS_IN_DIR = (
    'data/External/GiveDirectly/GE_HH_Census_2017-07-17_cleanGPS.csv')
CENSUS_MASTER_IN_DIR = (
    'data/External/GiveDirectly/GE_HH-Census_Analysis_RA_2017-07-17.dta')

ys = ['building_footprint',
      'tin_roof_area',
      'night_light']

_, df = collate_data(
    SAT_IN_DIR, SVY_IN_DIR,
    CENSUS_GPS_IN_DIR, CENSUS_MASTER_IN_DIR,
    NL_IN_DIR)

print('---- Results ----')

for y in ys:
    print(y, '\n')
    model = smf.ols(f'sat_{y} ~ in_treatment_group + hi_sat', data=df)
    result = model.fit()
    print(result.summary())
