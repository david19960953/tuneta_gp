# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:16:26 2022

@author: USER
"""

# from tuneta.tune_ta import TuneTA
import pandas as pd
# from pandas_ta import percent_return
# from sklearn.model_selection import train_test_split
# import yfinance as yf
# import pickle
import alphalens
import missingno
import numpy as np
# import pandas_ta as pta
import time
import os
# from tuneta.transTuneTa import *
import matplotlib.pyplot as plt
#開舊的X_需要設定一下

import pandas as pd
import gzip
import pickle
year = 2017
raw_X_ = pd.read_pickle(r'saveTransta/X_%s-12-31.pickle'%year)
# raw_X_ = pd.read_pickle(r'saveTransta/X_%s-06-30.pickle'%year)

# raw_X_ = pd.read_pickle(r'I:\我的雲端硬碟\colab_高曼_test\ETF_save/Xetf_%s-12-31.pickle'%year)
X = raw_X_.loc[:,'close'].to_frame()
# X.index.levels[0] = pd.to_datetime(X.index.levels[0])

'''
#  1~3月
train_start = '%s-01-01'%(year-3)
train_end = '%s-01-01'%(year)
# eval_ = end = '%s-07-31'%(year-1)
backtest_end = '%s-03-31'%(year)

# 4~6月
train_start = '%s-04-01'%(year-3)
train_end = '%s-04-01'%(year)
# eval_ = end = '%s-07-31'%(year-1)
backtest_end = '%s-06-30'%(year)


# 7~9月
train_start = '%s-07-01'%(year-4)
train_end = '%s-07-01'%(year-1)
# eval_ = end = '%s-07-31'%(year-1)
backtest_end = '%s-09-30'%(year-1)


# 10~12月
train_start = '%s-10-01'%(year-4)
train_end = '%s-10-01'%(year-1)
# eval_ = end = '%s-07-31'%(year-1)
backtest_end = '%s-12-31'%(year-1)


'''

# 4~6月
train_start = '%s-04-01'%(year-3)
train_end = '%s-04-01'%(year)
# eval_ = end = '%s-07-31'%(year-1)
backtest_end = '%s-06-30'%(year)

'''
catboost 分類器
'''
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm



# X_ = pd.concat([feature ,X ],axis = 1, join = 'inner')

X_ = raw_X_.iloc[:].copy()

'''
相對分類
'''
temp = dict(tuple(X_.groupby(level = 0)))
q_bins = 10
ll = []
for df in temp.values():
    df2 = pd.qcut(df['return'], q_bins,
                      labels=[i for i in range(q_bins)]) 
    df2.name = 'lable'
    df = pd.concat([df,df2],axis = 1)
    ll.append(df)
X_ = pd.concat(ll)
y = X_[['return','lable']]
X_ = X_.drop(columns=['return' , 'slope','open', 'high', 'low', 'close','lable'])




find_cat = X_.dropna().iloc[1,:].to_frame().T
cat_feature = list()
for col in find_cat.columns:
  if type(find_cat[col].values[0]) == str:
    cat_feature.append(col)

t0 = X_.index.levels[0][X_.index.levels[0]<train_end]
# te = X_.index.levels[0][(X_.index.levels[0]>train_end) & (X_.index.levels[0]<= eval_)]
t1 = X_.index.levels[0][(X_.index.levels[0]>=train_end) & (X_.index.levels[0]<=backtest_end)]


X_train , y_train,  = X_.loc[t0], y.loc[t0] # 2019~2021 當訓練要寫 2019~2022 @@
# X_eval , y_eval  = X_.loc[te], y.loc[te]  # 2019~2021 當測試直接2022
X_test , y_test  = X_.loc[t1], y.loc[t1]  # 2019~2021 當測試直接2022



'''
classification
'''
model_cb = CatBoostClassifier(task_type='GPU', iterations=4000, 
                      random_state = 2021, depth = 6)


model_cb.fit(X_train, y_train['lable'], plot=True, 
            cat_features = cat_feature,
            
            # eval_set=(X_eval, y_eval['lable']), 
            use_best_model=True)






y_pred = model_cb.predict(X_test)
y_pred = pd.DataFrame(y_pred , columns = ['predict'] , index = y_test.index)
y_new = pd.concat([y_test,y_pred] ,axis = 1)
predictive_factor = y_test['return']  #這個只是借用值，沒有用這個跑
pricing = X.close.loc[predictive_factor.index]
pricing  = pricing.unstack(level=1)

factor_data = alphalens.utils.get_clean_factor_and_forward_returns(predictive_factor, 
                                  pricing, 
                                  periods=( [1,3,5]),
                                  quantiles=2,
                                  )

y_pred_soft = model_cb.predict_proba(X_test) #生成機率
y_pred_soft = pd.DataFrame(y_pred_soft , index = y_test.index)      
# predict_value = y_pred_soft.dot(np.array(range(10))) #看訓練的時候是切成幾等分，這邊就沿用等分

# q_bins = 5
n0 = -(q_bins/2 -0.5)
predict_value = y_pred_soft.dot(np.array([n0 + i for i in range(q_bins)])) 
predict_value.name = 'predict_value'
y_pred_soft = pd.concat([y_pred_soft, predict_value], axis = 1)



temp = dict(tuple(y_pred_soft.groupby(level = 0)))
# q_bins = 5 #看訓練的時候是切成幾等分，這邊就沿用等分
ll = []
for df in temp.values():
    df2 = pd.qcut(df['predict_value'], q_bins,
                      labels=[i for i in range(q_bins)]) 
    df2.name = 'predict_rank'
    df = pd.concat([df,df2],axis = 1)
    ll.append(df)
y_pred_soft = pd.concat(ll)
y_new = pd.concat([y_new , factor_data, y_pred_soft] , axis = 1)


temp9 = []
temp0 =  []
groupby = 'predict_rank'
need = '3D'
weight = int(need[:-1])
see = dict(tuple(y_new.groupby(by = groupby)))
ret_df = pd.DataFrame()
for key in see:
    df =see[key]
    # df['predict_diff'] = abs(df['predict_rank'].astype(int) - df['predict'])
    # df = df.drop(index = df.loc[df['predict_diff']>2].index)  
    df = df.drop(index = df.loc[df[key]<0.15].index)  #可以看機率的大小
    if key ==9:temp9.append(df)
    if key ==0:temp0.append(df) 
    df = df[need].unstack(level=1)
    temp = df.apply(lambda x: x.mean(), axis = 1)
    temp.name = key
    ret_df = pd.concat( [ret_df , temp],axis = 1)

ret_df = ret_df.applymap(lambda x:x/weight)
ret_df['alpha0'] = (ret_df[9] - ret_df[0])/2
# ret_df['alpha31'] = ret_df[3] - ret_df[1]
# ret_df['alpha01'] = (ret_df['alpha0'] + ret_df['alpha1'])/2
ret_df.fillna(0,inplace = True)
# ret_df[[0,1,3,4,'alpha40','alpha31']].cumsum().plot()
ret_df[[0,1,7,8,9,'alpha0']].cumsum().plot()
# ret_df.cumsum().plot()











