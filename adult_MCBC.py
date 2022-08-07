# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:46:28 2022

@author: Fred
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from time import time
import xgboost
from sklearn.metrics import roc_auc_score, cohen_kappa_score


df1 = pd.read_csv("adult.data", header = 0, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',  'capital-loss', 'hours-per-week', 'native-country', 'class'])
df2 = pd.read_csv("adult.test", header = 0, names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',  'capital-loss', 'hours-per-week', 'native-country', 'class'])
df1['Y'] = np.where(df1['class']==' >50K', 1, 0)
df2['Y'] = np.where(df2['class']==' >50K.', 1, 0)

df1['marker'] = 1
df2['marker'] = 2

df = df1.append(df2)


df = df[df['native-country']!= ' Holand-Netherlands']

df['marital-status'] = df['marital-status'].astype('category').cat.codes
df['occupation'] = df['occupation'].astype('category').cat.codes
df['relationship'] = df['relationship'].astype('category').cat.codes
df['race'] = df['race'].astype('category').cat.codes
df['sex'] = df['sex'].astype('category').cat.codes
df['native-country'] = df['native-country'].astype('category').cat.codes
df['workclass'] = df['workclass'].astype('category').cat.codes

df1 = df[df['marker']==1]
df2 = df[df['marker']==2]

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
    return params, F,iter_store, F_store




entropy_params = {
    'eta': [0.0, 1],
    'max_depth': [1, 50],
    'subsample': [0,1],
    'colsample_bytree': [0,1]
    }

mod = xgboost.XGBClassifier()

restrictions = ['I', 'I']

x_train= df1.drop(['Y', 'class', 'education'], axis = 1)
y_train=df1['Y']


entropy_params, entropy_score, iters, F = mcm_halving(x_train, y_train, "accuracy", mod, entropy_params, restrictions, .05)




