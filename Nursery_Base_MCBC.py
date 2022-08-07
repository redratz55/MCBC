# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:03:58 2022

@author: Fred
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
from skopt import BayesSearchCV
from skopt.space import Space
from skopt.sampler import Lhs



names = ["t"+str(i) for i in range(9)]
df = pd.read_csv("nursery.data", index_col=False, skiprows = 1, names = names)


#Encode Categorical
for col in df.columns:
    if df[col].dtype=='O':
        df[col] = df[col].astype('category').cat.codes
    
#Seeded random numbers   
x_train, x_test, y_train, y_test = train_test_split(df.drop('t8',  axis = 1), df['t8'], random_state = 2)




#Algorithm with restricting upper and lower bounds
def mcm_opt(X, Y, metric, model, params, restrictions, cv=10, I=10, J =6, seed = 1):
    '''
    

    Parameters
    ----------
    X : Array or Pandas DF
        Independent Variables.
    Y : Array or Pandas Series
        Dependent Variables.
    metric : String
        Scoring Metric.
    model : Sklearn Model Object
        Model whose parameters we are tuning.
    params : Dictionary
        Dictionary of tuning parameters, with list of [lower_limit, Interval_width].
    restrictions : List of strings
        Dictionary of parameter restrictions valid inputs:
            I = Integer
            F = Float
            
    cv : Int, optional
        Number of K-fold crossvalidations to conduct. The default is 10.
    I : Int, optional
        Number of iterations to conduct at each half. The default is 10.
    J : Int, optional
        Number of Halves to conduct. The default is 6.
    seed : Int, optional
        The random number seed for numpy. The default is 1.

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
    rng = np.random.default_rng(seed)
    upper_limit = []
    lower_limit = []
    for key in params:
        upper_limit.append(params[key][1])
        lower_limit.append(params[key][0])
    min_ll = [i for i in lower_limit]
    max_UL = [i for i in upper_limit]
    
    output_params = {key: [] for key in params.keys()}
    test_params = {key: None for key in params.keys()}
    
    A = [max_UL[i]/2 for i in range(len(upper_limit))]
    F = 0
    pre_checked = []
    j = 1
    while True:
        store_scores = []
        store_values = []
            
        print(f'Sampling {I} iterations at current halving')
        for i in range(I):

            T = []
            
            for k in range(len(upper_limit)):
            
                T.append( lower_limit[k] + (max_UL[k]/2**(j-1))*rng.uniform(0,1))
            
           
                if restrictions[k] == 'I':
                    T[k] = int(T[k])
                    
                if T[k] > max_UL[k]:
                    T[k] = max_UL[k]
                if T[k] < min_ll[k]:
                    T[k] = min_ll[k]
                    
            count = 0
            for key in test_params.keys():
                test_params[key] = T[count]
                count += 1

            if T not in pre_checked:
                pre_checked.append(T)
                
        
                score = cross_val_score(model.set_params(**test_params), X = X, y = Y, scoring = metric, cv = cv)
                store_scores.append(np.mean(score))
                store_values.append(T)
            
            
                if np.mean(score) > F:
                    
                    F = np.mean(score)
                    A = T
                    print(f'New Best Score: {F}')
                    print(f'New Best Params: {A}')
                    for k in range(len(upper_limit)):       
                        
                        lower_limit[k] = A[k] - max_UL[k]/2**(j)
                        upper_limit[k] = A[k] + max_UL[k]/2**(j)
                    
                       
                        if lower_limit[k] < min_ll[k]:
                            lower_limit[k] = min_ll[k]
                            
                        if upper_limit[k] > max_UL[k]:
                            upper_limit[k] = max_UL[k]
                    print(f'New Center of Search: {upper_limit}, {lower_limit}')
       
        #Increase power, shrink bounds 
        j+=1   
        for k in range(len(upper_limit)):       
            
           
            
            
            lower_limit[k] = A[k] - max_UL[k]/2**(j)
            upper_limit[k] = A[k] + max_UL[k]/2**(j)
        
           
            if lower_limit[k] < min_ll[k]:
                lower_limit[k] = min_ll[k]
            if upper_limit[k] > max_UL[k]:
                upper_limit[k] = max_UL[k]

        print(f'New Center of Search: {upper_limit}, {lower_limit}')
        print(f'Current Best Score: {F}')

         

           
        if j == J:
            print('Minimum Region Reached, Terminating')
            second_time = time()
            print(f'Total Run Time: {second_time - first_time}')
            print(A, F)
            count = 0
            for key in output_params.keys():
                output_params[key] = A[count]
                count += 1
            return output_params, F, second_time - first_time

            
        
    second_time = time()
    print(f'Total Run Time: {second_time - first_time}')
    print(A, F)
    count = 0
    for key in output_params.keys():
        output_params[key] = A[count]
        count += 1
    return output_params, F, second_time - first_time




"""
Sample 10 iterataions of mcm
"""

mcm_score = []
mcm_time = []
mcm_train_score = []
seeds = [i for i in range(1,11)]

for i in range(10):
    params = {'n_estimators': [1, 200],
              'max_depth': [1, 20]}

    restrictions = ['I', 'I']
    
    out_params, out_score, tot_time = mcm_opt(x_train, y_train, 'accuracy', RF(random_state = seeds[i]), params, restrictions, seed = seeds[i])
    mcm_train_score.append(out_score)
    mcm_time.append(tot_time)
    
    model = RF(random_state = seeds[i])
    model.set_params(**out_params)
    
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    
    mcm_score.append(accuracy_score(y_test, pred))
    print(f'\n\nPrevious score: {mcm_score[-1]}, AVG Score: {np.mean(mcm_score)}, Deviation: {np.std(mcm_score)}, AVG Time: {np.mean(mcm_time)}\n\n')

print(f'Training Score: {np.mean(mcm_train_score)}, train_std: {np.std(mcm_train_score)}')


"""
Sample 10 iterations Bayes CV
"""
bayes_scores = []
bayes_time = []



for i in range(10):
    
    param_grid = {
        'n_estimators': [1, 200],
        'max_depth': [1, 20]
        
        }
    
    bayes_search = BayesSearchCV(estimator=RF(random_state = seeds[i]), search_spaces=param_grid,  cv=10, verbose=1, random_state = seeds[i])
    start_time = time()
    bayes_search.fit(x_train, y_train)
    end_time = time()
    
    bayes_time.append(end_time-start_time)
    
    mod = bayes_search.best_estimator_
    mod.fit(x_train, y_train)
    
    pred = mod.predict(x_test)
    
    
    bayes_scores.append(accuracy_score(y_test, pred))
    print(f'\n Previous Score: {bayes_scores[-1]}, Current AVG: {np.mean(bayes_scores)}, Deviation: {np.std(bayes_scores)}, AVG Time: {np.mean(bayes_time)}\n')


print(f'\n\nMCM Average Score: {np.mean(mcm_score)}, Deviation: {np.std(mcm_score)}, AVG Time: {np.mean(mcm_time)}\n')

print(f'Bayes AVG Score: {np.mean(bayes_scores)}, Deviation: {np.std(bayes_scores)}, AVG Time: {np.mean(bayes_time)}')


    










