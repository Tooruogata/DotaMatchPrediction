# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:10:56 2021

@author: Tooru Ogata y Jhosua Torres
proyect: Predicción de resultados de partidas de dota2

source: https://www.opendota.com/explorer
"""

import pandas as pd
import os
import numpy as np

path = r'D:\Otros Proyectos\Dota2\data'
os.chdir(path)

year_0 = 2018
year_1 = 2021

root_filename = 'query2_dota2_'

'''
1. Abriendo la data
'''

team_name = pd.read_csv('dota2_team.csv')
team_name.columns = ['team_id','team_name', 'team_tag']

df_append = pd.DataFrame()
for year in range(year_0,year_1+1):
    filename = str(root_filename) + str(year) + r'.csv'
    df = pd.read_csv(filename)
    df['year'] = year
    df_append = df_append.append(df)
    del df    

df_append['team_id'] = np.where(df_append['radiant_win'] == df_append['win'], df_append['radiant_team_id'], df_append['dire_team_id'])
df_append.columns

df_append = pd.merge(df_append, team_name, left_on=['team_id'] , right_on=['team_id'], how='left')

        
'''
2. Feature Engineering
'''    

df_append.columns

#Maestro de teams
df_teams = df_append[['year','match_id','team_name','win']]
df_teams = pd.get_dummies(df_teams, prefix='win', columns=['win'])
df_teams = df_teams.dropna()
df_teams = df_teams.drop_duplicates()

#Creación de dummies
role_list = ['Support','Nuker','Initiator','Escape','Durable','Disabler','Carry','Jungler','Pusher'  ]

for rol in role_list:
    df_append[rol] = df_append['roles'].str.contains(rol).astype(int)

df_append = pd.get_dummies(df_append, prefix='attribute', columns=['primary_attr'])
df_append = pd.get_dummies(df_append, prefix='attack', columns=['attack_type'])

#Group by player/hero last 5-10 matches
df_player = df_append
df_player = df_player.drop(labels=["team_name"], axis=1)
df_player = df_player.groupby(['win','year','match_id','start_time','account_id','hero_id','name'], as_index=False).agg( duration=('duration','mean'), 
                                                                                    sum_support=('Support','sum'), 
                                                                                    sum_nuker=('Nuker','sum'), 
                                                                                    sum_initiator=('Initiator','sum'), 
                                                                                    sum_escape=('Escape','sum'), 
                                                                                    sum_durable=('Durable','sum'), 
                                                                                    sum_disabler=('Disabler','sum'), 
                                                                                    sum_carry=('Carry','sum'), 
                                                                                    sum_jungler=('Jungler','sum'), 
                                                                                    sum_pusher=('Pusher','sum'), 
                                                                                    sum_agi=('attribute_agi','sum'), 
                                                                                    sum_int=('attribute_int','sum'), 
                                                                                    sum_str=('attribute_str','sum'), 
                                                                                    sum_melee=('attack_Melee','sum'), 
                                                                                    sum_ranged=('attack_Ranged','sum'), 
                                                                                    sum_kills=('kills','sum'), 
                                                                                    sum_deaths= ('deaths','sum'),
                                                                                    sum_assists= ('assists','sum'),
                                                                                    mean_lasthits= ('last_hits','mean'),
                                                                                    mean_denies= ('denies','mean'),
                                                                                    sum_observers= ('observers_placed','sum'),
                                                                                    sum_towers= ('towers_killed','sum'),
                                                                                    mean_gold_min=('gold_per_min','mean'),
                                                                                    mean_exp_min=('xp_per_min','mean'))





df_append.columns

#Group by sum y mean a nivel de match
df_append = df_append.groupby(['win','year','match_id','start_time'], as_index=False).agg( duration=('duration','mean'), 
                                                                                    sum_support=('Support','sum'), 
                                                                                    sum_nuker=('Nuker','sum'), 
                                                                                    sum_initiator=('Initiator','sum'), 
                                                                                    sum_escape=('Escape','sum'), 
                                                                                    sum_durable=('Durable','sum'), 
                                                                                    sum_disabler=('Disabler','sum'), 
                                                                                    sum_carry=('Carry','sum'), 
                                                                                    sum_jungler=('Jungler','sum'), 
                                                                                    sum_pusher=('Pusher','sum'), 
                                                                                    sum_agi=('attribute_agi','sum'), 
                                                                                    sum_int=('attribute_int','sum'), 
                                                                                    sum_str=('attribute_str','sum'), 
                                                                                    sum_melee=('attack_Melee','sum'), 
                                                                                    sum_ranged=('attack_Ranged','sum'), 
                                                                                    sum_kills=('kills','sum'), 
                                                                                    sum_deaths= ('deaths','sum'),
                                                                                    sum_assists= ('assists','sum'),
                                                                                    mean_lasthits= ('last_hits','mean'),
                                                                                    mean_denies= ('denies','mean'),
                                                                                    sum_observers= ('observers_placed','sum'),
                                                                                    sum_towers= ('towers_killed','sum'),
                                                                                    mean_gold_min=('gold_per_min','mean'),
                                                                                    mean_exp_min=('xp_per_min','mean'))

#Porcentaje de roles
df_append['sum_roles'] =    df_append['sum_support'] + df_append['sum_nuker'] + df_append['sum_initiator'] + df_append['sum_escape'] + df_append['sum_durable'] + df_append['sum_disabler'] + df_append['sum_carry'] + df_append['sum_jungler'] + df_append['sum_pusher'] 

df_append['sum_support'] = df_append['sum_support']/df_append['sum_roles'] 
df_append['sum_nuker'] = df_append['sum_nuker']/df_append['sum_roles'] 
df_append['sum_initiator'] = df_append['sum_initiator']/df_append['sum_roles'] 
df_append['sum_escape'] = df_append['sum_escape']/df_append['sum_roles'] 
df_append['sum_durable'] = df_append['sum_durable']/df_append['sum_roles'] 
df_append['sum_disabler'] = df_append['sum_disabler']/df_append['sum_roles'] 
df_append['sum_carry'] = df_append['sum_carry']/df_append['sum_roles'] 
df_append['sum_jungler'] = df_append['sum_jungler']/df_append['sum_roles'] 
df_append['sum_pusher'] = df_append['sum_pusher']/df_append['sum_roles'] 

#Porcentaje primary attribute
df_append['sum_agi'] = df_append['sum_agi']/5
df_append['sum_int'] = df_append['sum_int']/5
df_append['sum_str'] = df_append['sum_str']/5

#Porcentaje melee/ranged
df_append['sum_melee'] = df_append['sum_melee']/5
df_append['sum_ranged'] = df_append['sum_ranged']/5

#Target dummy result win and lose
df_append = pd.get_dummies(df_append, prefix='win', columns=['win'])

#Merge to teams first non missing
df_append = pd.merge(df_append, df_teams, left_on=['year','match_id', 'win_False', 'win_True'] , right_on=['year','match_id', 'win_False', 'win_True'], how='left')
del df_teams

df_append.to_csv('data_consolidada.csv')

'''
3. Models 1
'''    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_append = pd.read_csv('data_consolidada.csv')

x = df_append[['duration', 'sum_support',
       'sum_nuker', 'sum_initiator', 'sum_escape', 'sum_durable',
       'sum_disabler', 'sum_carry', 'sum_jungler', 'sum_pusher', 'sum_agi',
       'sum_int', 'sum_str', 'sum_melee', 'sum_ranged', 'sum_kills',
       'sum_deaths', 'sum_assists', 'mean_lasthits', 'mean_denies',
       'sum_observers', 'sum_towers', 'mean_gold_min', 'mean_exp_min']]
y = df_append['win_True']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

print(model.coef_)
print(model.intercept_)

pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])

predictions = model.predict(x_test)

plt.scatter(y_test, predictions)
plt.hist(y_test - predictions)

from sklearn import metrics

print('MAE')
print(metrics.mean_absolute_error(y_test, predictions))
print('MAE')
print(metrics.mean_squared_error(y_test, predictions))
print('root-MSE')
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

'''
4. Models 2 - a
'''

import pandas as pd
import re

df_append = pd.read_csv('data_consolidada.csv')
df_append = df_append.dropna()

df_append['team_name'] = df_append['team_name'].map(lambda x: re.sub(r'[^a-zA-Z0-9\._-]', '', x))

df_append = pd.get_dummies(df_append, prefix='team', columns=['team_name'])

df_2021 = df_append.query('(year == 2021)')
df_append = df_append.query('(year != 2021)')

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df_append[['sum_support',
       'sum_nuker', 'sum_initiator', 'sum_escape', 'sum_durable',
       'sum_disabler', 'sum_carry', 'sum_jungler', 'sum_pusher', 'sum_agi',
       'sum_int', 'sum_str', 'sum_melee', 'sum_ranged'
       ]]
Y = df_append['win_True']

max_depth = [6,8,10]
reg_alpha = [0.5,1.5,2.5]
reg_lambda = [1.5,2.5,3.5]
n_estimators = [100,300,500]

for rdepth in max_depth:
    for ralpha in reg_alpha:
        for rlambda in reg_lambda:
            for restimator in n_estimators:

                seed = 7
                test_size = 0.33
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
                
                # fit model no training data
                model = XGBClassifier(gamma=0, 
                                      learning_rate=0.300000012,
                                      max_depth=rdepth,
                                      n_estimators=restimator, 
                                      reg_alpha=ralpha, 
                                      reg_lambda=rlambda, 
                                      scale_pos_weight=1, subsample=1,
                                      tree_method='exact', validate_parameters=1, verbosity=None, 
                                      eval_metric='logloss',
                                      objective = 'binary:logistic')
                model.fit(X_train, y_train)
                
                print('max_depth - ' + str(rdepth))
                print('reg_alpha - ' + str(ralpha))
                print('reg_lambda - ' + str(rlambda))
                print('n_estimators - ' + str(restimator))
                #print(model)
                
                # make predictions for test data
                y_pred = model.predict(X_test)
                predictions = [round(value) for value in y_pred]
                
                # evaluate predictions
                accuracy = accuracy_score(y_test, predictions)
                print("Test - Accuracy: %.2f%%" % (accuracy * 100.0))
                
                '''
                4. Models 2 - b
                '''
                # Predict in 2021
                X = df_2021[['sum_support',
                       'sum_nuker', 'sum_initiator', 'sum_escape', 'sum_durable',
                       'sum_disabler', 'sum_carry', 'sum_jungler', 'sum_pusher', 'sum_agi',
                       'sum_int', 'sum_str', 'sum_melee', 'sum_ranged', 
                       ]]
                Y = df_2021['win_True']
                
                # make predictions for test data
                y_pred = model.predict(X)
                predictions = [round(value) for value in y_pred]
                
                # evaluate predictions
                accuracy = accuracy_score(Y, predictions)
                print("Validation - Accuracy: %.2f%%" % (accuracy * 100.0))


import pickle

filename = 'dota2_model'
pickle.dump(model, open(filename, 'wb'))