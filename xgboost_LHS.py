# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 04:38:57 2022

@author: Fred
"""

from MCBS import mcm_opt_ucb
from sklearn.metrics import mean_absolute_percentage_error
from pmlb import fetch_data
from xgboost import XGBRegressor as XGR
import numpy as np
from sklearn.model_selection import train_test_split
from time import time


datasets = ['503_wind', '529_pollen', '547_no2', '560_bodyfat', '1030_ERA']
seed = 1

mcbc_scores = []
mcbc_times = []

for dset in datasets:
    X, y = fetch_data(dset, return_X_y=True)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = seed)
    m_s = []
    m_t = []
    b_s = []
    b_t = []
    
    for s in range(2, 12):
        seed += s
        print(f"Current seed: {seed}")
        params = {'eta': [0.0, 1.0], 'gamma': [0.0, 10.0], 'max_depth': [0, 20], 'min_child_weight': [0, 20], 'max_delta_step': [0, 20], 'subsample': [0.0, 1.0]}
        restrictions = ['R', 'R', 'I', 'I', 'I', 'R']
        out_params, out_score, tot_time, bseed = mcm_opt_ucb(x_train, y_train, 'neg_mean_absolute_percentage_error', XGR(random_state = seed),  params, restrictions,seed =  seed, r_yes = True, debug = False, maxi = False, lhs_sampler = True)
        m_t.append(tot_time)
        out_params['random_state'] = bseed
        
        model = XGR(**out_params)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        m_s.append(mean_absolute_percentage_error(y_test, pred))
        print(f'Current Perf: MCBC: {np.mean(m_s)}, {np.std(m_s)}, {np.mean(m_t)}')
    mcbc_scores.append([np.mean(m_s), np.std(m_s)])
    mcbc_times.append(np.mean(m_t))
    


print('Final Performance:')
print(f'MCBC, Score: {mcbc_scores}, Time: {mcbc_times}')

