# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:58:00 2022

@author: Fred
"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from skopt.space import Space
from skopt.sampler import Lhs



def mcm_opt_ucb(X, Y, metric, estimator, params, restrictions, cv=10, I=10, J =6, seed = 1, restrict_upper = True, restrict_lower = True, maxi=True, lhs_sampler = False,r_yes = True, debug = False):
    '''
    

    Parameters
    ----------
    X : Array or Pandas DF
        Independent Variables.
    Y : Array or Pandas Series
        Dependent Variables.
    metric : String
        Scoring Metric.
    estimator : Sklearn Model Object
        Model whose parameters we are tuning.
    params : Dictionary
        Dictionary of tuning parameters, with list of [lower_limit, Interval_width].
    restrictions : List of strings
        Dictionary of parameter restrictions valid inputs:
            I = Integer
            F = Float
            IB = Integer bounded
            FB = Float bounded.
    cv : Int, optional
        Number of K-fold crossvalidations to conduct. The default is 10.
    I : Int, optional
        Number of iterations to conduct at each half. The default is 10.
    J : Int, optional
        Number of Halves to conduct. The default is 6.
    prior_mu : Float, optional
        The initial prior mean. The default is 0.
    prior_sigma : Float, optional
        The initial prior standard deviation. The default is 1.
    seed : Int, optional
        The random numpy seed for numpy. The default is 1
    

    Returns
    -------
    output_params : Dictionary
        DDictionary of best found params.
    F : Float
        Score of best found params on metric.
    Time: Float
        Total Training Time in seconds.

    '''
    
    first_time = time()
    upper_limit = []
    lower_limit = []
    for key in params:
        upper_limit.append(params[key][1])
        lower_limit.append(params[key][0])
    min_ll = [i for i in lower_limit]
    max_UL = [i for i in upper_limit]
    
    output_params = {key: [] for key in params.keys()}
    
    
    pre_checked = []
    j = 1
    random_seed = 1
    param_keys = [i for i in params.keys()]
    if r_yes:
        param_keys.append('random_state')
    
    
    A = [max_UL[i]/2 for i in range(len(upper_limit))]
    
    if maxi:
        F = 0
    if not maxi:
        F = -99999999999999

    best_seed = seed
    counts = 0
    while True:
        
        print(f'Sampling {I} iterations at current halving')
        store_values = []

       
        if  lhs_sampler:
            try:
                space = Space(list(zip(lower_limit, upper_limit)))
                T = Lhs(lhs_type = 'centered').generate(space, 10, random_state = seed)
            except ValueError:
                print("Minimum Interval Width, Terminating")
                second_time = time()
                count = 0
                for key in output_params.keys():
                    output_params[key] = A[count]
                    count += 1

                return output_params, F, second_time - first_time, best_seed

            for t in T:
                random_seed += 1
                if r_yes:
                    t.append(seed)
                test_params = dict(zip(param_keys, t))
    
                if t not in pre_checked:
                    pre_checked.append(t)
            
                    score = cross_val_score(estimator.set_params(**test_params), X = X, y = Y, scoring = metric, cv = cv)
                    store_values.append(t)
    
                
                    if (np.mean(score) > F) or (j==1 and counts == 0):
                        
                        F = np.mean(score)
                        A = t
                        if r_yes:
                            A = A[:-1]
                        best_seed = seed
                        if debug:
                            print(f'New Best Score: {F}')
                            print(f'New Best Params: {A}')
                        for k in range(len(upper_limit)):
                            if restrictions[k] == 'I':
                                lower_limit[k] = int(A[k] - max_UL[k]/2**(j))
                                upper_limit[k] = int(A[k] + max_UL[k]/2**(j))
                            else:
                                lower_limit[k] = A[k] - max_UL[k]/2**(j)
                                upper_limit[k] = A[k] + max_UL[k]/2**(j)
                                
                        
                           
                            if lower_limit[k] < min_ll[k] and restrict_lower:
                                lower_limit[k] = min_ll[k]
                            if upper_limit[k] > max_UL[k] and restrict_upper:
                                upper_limit[k] = max_UL[k]
                counts += 1

      
        if not lhs_sampler:
            for i in range(I):
                try:
                    space = Space(list(zip(lower_limit, upper_limit)))
                    T = space.rvs(1, random_state = random_seed)[0]

                except ValueError:
                    print("Minimum Interval Width, Terminating")
                    second_time = time()
                    count = 0
                    for key in output_params.keys():
                        output_params[key] = A[count]
                        count += 1

                    return output_params, F, second_time - first_time, best_seed
                random_seed += 1
                if r_yes:
                    T.append(seed)
                test_params = dict(zip(param_keys, T))
    
                if T not in pre_checked:
                    pre_checked.append(T)
            
                    score = cross_val_score(estimator.set_params(**test_params), X = X, y = Y, scoring = metric, cv = cv)
                    store_values.append(T)
    
                
                    if np.mean(score) > F:
                        
                        F = np.mean(score)
                        A = T
                        if r_yes:
                            A = A[:-1]
                        best_seed = seed
                        if debug:
                            print(f'New Best Score: {F}')
                            print(f'New Best Params: {A}')
                        
                        for k in range(len(upper_limit)):
                            if restrictions[k] == 'I':
                                lower_limit[k] = int(A[k] - max_UL[k]/2**(j))
                                upper_limit[k] = int(A[k] + max_UL[k]/2**(j))
                            else:
                                lower_limit[k] = A[k] - max_UL[k]/2**(j)
                                upper_limit[k] = A[k] + max_UL[k]/2**(j)
                                
                        
                           
                            if lower_limit[k] < min_ll[k] and restrict_lower:
                                lower_limit[k] = min_ll[k]
                            if upper_limit[k] > max_UL[k] and restrict_upper:
                                upper_limit[k] = max_UL[k]


        j+=1   
        if j == J:
            print('Minimum Region Reached, Terminating')
            second_time = time()
            print(f'Total Run Time: {second_time - first_time}')
            print(A, F)
            count = 0
            for key in output_params.keys():
                output_params[key] = A[count]
                count += 1
            return output_params, F, second_time - first_time, best_seed
        
            
        for k in range(len(upper_limit)):
            if restrictions[k] == 'I':
                lower_limit[k] = int(A[k] - max_UL[k]/2**(j))
                upper_limit[k] = int(A[k] + max_UL[k]/2**(j))
            else:
                lower_limit[k] = A[k] - max_UL[k]/2**(j)
                upper_limit[k] = A[k] + max_UL[k]/2**(j)
        
           
            if lower_limit[k] < min_ll[k] and restrict_lower:
                lower_limit[k] = min_ll[k]
            if upper_limit[k] > max_UL[k] and restrict_upper:
                upper_limit[k] = max_UL[k]
   
            
        if debug:    
            print(f'New Center of Search: {upper_limit}, {lower_limit}')
        
        

            
        
    second_time = time()
    print(f'Total Run Time: {second_time - first_time}')
    print(A, F)
    count = 0
    for key in output_params.keys():
        output_params[key] = A[count]
        count += 1
    return output_params, F, second_time - first_time, best_seed








