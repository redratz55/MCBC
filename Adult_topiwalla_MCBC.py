# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:22:11 2022

@author: Fred
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import xgboost
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score




df = pd.read_csv("adult_data_topiwalla.csv")
cols = df.columns[1:]
df = df[cols]
df = df[df['country']!= ' Holand-Netherlands']

df['marital'] = df['marital'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['relationship'] = df['relationship'].astype('category').cat.codes
df['race'] = df['race'].astype('category').cat.codes
df['sex'] = df['sex'].astype('category').cat.codes
df['country'] = df['country'].astype('category').cat.codes
df['type_employer'] = df['type_employer'].astype('category').cat.codes

df['Y'] = np.where(df['income']=='>50K', 1, 0)


df1, df2, y_train, y_test = train_test_split(df.drop(['Y',  'education', 'income'], axis = 1), df['Y'], test_size = 0.3, shuffle=False)

df1['Y'] = y_train
df2['Y'] = y_test



"""
Entropy Halving Algorithm
"""

def mcm_halving(X, Y, metric, model, params, restrictions, alpha):
    first_time = time()
    upper_limit = []
    lower_limit = []
    iter_store = []
    F_store = []
    for key in params:
        upper_limit.append(params[key][1])
        lower_limit.append(params[key][0])
    min_ll = [i for i in lower_limit]
    max_UL = [i for i in upper_limit]
    
    A = 5
    F = 0
    
    for j in range(1,7):
        store_scores = []
        store_values = []
        for i in range(300):
            if i%10 == 0:
                print(f'{(i/300)*100}% Complete with {j} Halving')
            T = []
            
            for k in range(len(upper_limit)):
            
                T.append( lower_limit[k] + upper_limit[k]*np.random.uniform(0,1))
            
            T[1] = int(T[1])

            if T[0]> 1:
                T[0] = 1
            if T[2] >1:
                T[2]=1
            if T[3]>1:
                T[3] = 1
                
            model = xgboost.XGBClassifier(eval_metric= 'logloss', use_label_encoder = False,  eta = T[0], max_depth = T[1], subsample = T[2], colsample_bytree = T[3])


            score = cross_val_score(model, X = X, y = Y, scoring = metric, cv = 10)
            store_scores.append(np.mean(score))
            store_values.append(T)
            
            if np.mean(score) > F:
                F = np.mean(score)
                A = T
                iter_store.append(i+(100*(j-1)))
                F_store.append(F)
                print(f'New Best Score: {F}')
                print(f'New Best Params: {A}')
                print(f'Iteration: {i+(100*(j-1))}')

                
            
                
        for k in range(len(upper_limit)):       
            upper_limit[k] = A[k] + max_UL[k]/2**(j)
            lower_limit[k] = A[k] - max_UL[k]/2**(j)
        
           
            if lower_limit[k] < min_ll[k]:
                lower_limit[k] = min_ll[k]
                
        if upper_limit[0] > 1:
            upper_limit[0] = 1
        if upper_limit[2]>1:
            upper_limit[2] = 1
        if upper_limit[3]>1:
            upper_limit[3]=1
        print("Iteration Complete Decreasing Width")
        
    second_time = time()
    print(f'Total Run Time: {second_time - first_time}')
    print(A, F)
    count = 0
    for key in params.keys():
        params[key] = A[count]
        count += 1
    return params, F, iter_store, F_store




entropy_params = {
    'eta': [0.0, 1],
    'max_depth': [1, 50],
    'subsample': [0,1],
    'colsample_bytree': [0,1]
    }

mod = xgboost.XGBClassifier()

restrictions = ['I', 'I']

x_train= df1.drop(['Y'], axis = 1)
y_train=df1['Y']


entropy_params, entropy_score, iters, F = mcm_halving(x_train, y_train, "accuracy", mod, entropy_params, restrictions, .05)



T = [0.30495434924159376, 6, 0.9324184987313944, 0.1646852547824693]
mod = xgboost.XGBClassifier(eval_metric= 'logloss', use_label_encoder = False,  eta = T[0], max_depth = T[1], subsample = T[2], colsample_bytree = T[3])

mod.fit(x_train, y_train)

pred = mod.predict(df2.drop(['Y'], axis = 1))
proba = mod.predict_proba(df2.drop(['Y'], axis = 1))[:,1]

y_test = df2['Y']


accuracy_score(y_test, pred)
roc_auc_score(y_test, proba)
cohen_kappa_score(y_test, pred)












