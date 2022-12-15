# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:13:52 2022

@author: USER
"""


import pandas as pd
import pickle
import alphalens
import missingno
import numpy as np

import matplotlib.pyplot as plt
import datetime
import quantstats as qs
import time
import os

#開舊的X_需要設定一下


from catboost import CatBoostClassifier
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm


class UseCatBoost():
    def __init__(self, train_start, train_end, backtest_end,
                 target_col  = 'return',q_bins = 10):
        
        self.train_start = train_start
        self.train_end = train_end
        self.backtest_end = backtest_end
        self.target_col = target_col
        self.q_bins = q_bins


    def PrepareTraining(self, raw_X_):
        X_ = raw_X_.iloc[:].copy()
        self.close = raw_X_.loc[:,'close'].to_frame()
        temp = dict(tuple(X_.groupby(level = 0)))
        # self.q_bins = 10
        ll = []
        for df in temp.values():
            df2 = pd.qcut(df[self.target_col], self.q_bins,
                              labels=[i for i in range(self.q_bins)]) 
            df2.name = 'lable'
            df = pd.concat([df,df2],axis = 1)
            ll.append(df)
        X_ = pd.concat(ll)
        y = X_[[self.target_col,'lable']]
        if 'slope' in X_.columns:
             X_ = X_.drop(columns=['slope'])
         
        X_ = X_.drop(columns=['return' ,'open', 'high', 'low', 'close','volume','lable'])
        
        
        find_cat = X_.dropna().iloc[1,:].to_frame().T
        cat_feature = list()
        for col in find_cat.columns:
          if type(find_cat[col].values[0]) == str:
            cat_feature.append(col)
        self.cat_feature = cat_feature
        t0 = X_.index.levels[0][(X_.index.levels[0]<self.train_end) & (X_.index.levels[0]>= self.train_start)]
        # te = X_.index.levels[0][(X_.index.levels[0]>train_end) & (X_.index.levels[0]<= eval_)]
        t1 = X_.index.levels[0][(X_.index.levels[0]>=self.train_end) & (X_.index.levels[0]< self.backtest_end)]
        
        
        X_train ,y_train,  = X_.loc[t0], y.loc[t0] # 2019~2021 當訓練要寫 2019~2022 @@
        # X_eval , y_eval  = X_.loc[te], y.loc[te]  # 2019~2021 當測試直接2022
        X_test , y_test  = X_.loc[t1], y.loc[t1]  # 2019~2021 當測試直接2022
        return X_train, y_train, X_test, y_test


    def TrainClassify(self, iterations, random_state, verbose):
        model_cb = CatBoostClassifier(task_type='GPU', iterations= iterations, 
                              random_state = random_state , depth = 6, verbose = verbose)
        
        
        model_cb.fit(self.X_train, self.y_train['lable'], plot=True, 
                    cat_features = self.cat_feature,
                     verbose = verbose
                    # eval_set=(X_eval, y_eval['lable']), 
                    use_best_model=True)
        return model_cb


    def get_model(self, raw_X_, iterations = 4000, random_state = 2021, verbose = True):
        self.X_train, self.y_train, self.X_test, self.y_test = self.PrepareTraining(raw_X_)
        self.model_cb = self.TrainClassify(iterations, random_state = random_state, verbose = verbose)
        


    def get_predict(self, test = True):
        if test == True:
            X_t = self.X_test
            y_t = self.y_test
        else:
            X_t = self.X_train
            y_t = self.y_train        
            
            
        y_pred = self.model_cb.predict(X_t)
        y_pred = pd.DataFrame(y_pred , columns = ['predict'] , index = y_t.index)
        y_new = pd.concat([y_t,y_pred] ,axis = 1)
        predictive_factor = y_t['return']  #這個只是借用值，沒有用這個跑
        pricing = self.close.loc[predictive_factor.index]
        pricing  = pricing.unstack(level=1)
        pricing.columns = pricing.columns.levels[1]
        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(predictive_factor, 
                                          pricing, 
                                          periods=( [1,2,3,4,5]),
                                          quantiles=3,
                                          )
        
        y_pred_soft = self.model_cb.predict_proba(X_t) #生成機率
        y_pred_soft = pd.DataFrame(y_pred_soft , index = y_t.index)      
        # predict_value = y_pred_soft.dot(np.array(range(10))) #看訓練的時候是切成幾等分，這邊就沿用等分
        
        n0 = -(self.q_bins/2 -0.5)
        predict_value = y_pred_soft.dot(np.array([n0 + i for i in range(self.q_bins)])) 
        predict_value.name = 'predict_value'
        y_pred_soft = pd.concat([y_pred_soft, predict_value], axis = 1)
        
        
        
        temp = dict(tuple(y_pred_soft.groupby(level = 0)))
        #看訓練的時候是切成幾等分，這邊就沿用等分
        ll = []
        for df in temp.values():
            df2 = pd.qcut(df['predict_value'], self.q_bins,
                              labels=[i for i in range(self.q_bins)]) 
            df2.name = 'predict_rank'
            df = pd.concat([df,df2],axis = 1)
            ll.append(df)
        y_pred_soft = pd.concat(ll)
        y_new = pd.concat([y_new , factor_data, y_pred_soft] , axis = 1)
        return y_new



    def naive_plot(self, y_new, need = '3D'):
        labels = [i for i in range(self.q_bins)]
        groupby = 'predict_rank' # ['predict_rank', 'predict]
        # need = '3D'
        weight = int(need[:-1])
        see = dict(tuple(y_new.groupby(by = groupby)))
        ret_df = pd.DataFrame()
        for key in see:
            df =see[key]
            df['predict_diff'] = abs(df['predict_rank'].astype(int) - df['predict'])
            df = df.drop(index = df.loc[df['predict_diff']>1].index)  
            df = df.drop(index = df.loc[df[key]<0.15].index)  #可以看機率的大小
            
            if key ==9:temp9 = df.iloc[:]
            if key ==0:temp0 =  df.iloc[:]
            df = df[need].unstack(level=1)
            temp = df.apply(lambda x: x.mean(), axis = 1)
            temp.name = key
            ret_df = pd.concat( [ret_df , temp],axis = 1)
        
        ret_df.index = pd.to_datetime(ret_df.index)
        ret_df = ret_df.sort_index()
        ret_df = ret_df.applymap(lambda x:x/weight)
        ret_df['day'] = range(len(ret_df))
        ret_df = ret_df.drop(index = ret_df.loc[ret_df['day']%weight != 0].index)
        
        ret_df.fillna(0,inplace = True)
        ret_df['alpha'] = (ret_df[labels[-1]] - ret_df[labels[0]])/2
        ret_df['alpha_leverage2'] = ret_df['alpha']*2
        # ret_df[[0,1,2,7,8,9]].cumsum().plot()
        ret_df[labels].cumsum().plot()
        ret_df[[labels[-1],labels[0],'alpha','alpha_leverage2']].cumsum().plot()



# year = 2016
# train_start = '%s-03-01'%(year-1)
# train_end = '%s-03-01'%(year)
# # eval_ = end = '%s-07-31'%(year-1)
# backtest_end = '%s-04-10'%(year)



# raw_X_ = pd.read_pickle(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta/X_%s-12-31.pickle'%year)
# ucb = UseCatBoost( train_start = train_start, train_end = train_end, backtest_end = backtest_end, 
#                  target_col  = 'return', q_bins = 10)

# ucb.get_model(raw_X_, iterations = 4000)
# y_new = ucb.get_predict(test = True)
# ucb.naive_plot( y_new, need = '3D')

    
if __name__ == '__main__':

    # from tuneta.tune_ta import TuneTA
    # transta = pd.read_pickle(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta\transta_2017-06-30.pickle')    

    import os 
    from dateutil.relativedelta import relativedelta
  
    list_dir = os.listdir(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta')
    list_dir = sorted([i for i in list_dir if 'X_' in i ]) #把2022 最後一個拿掉

    yNew_dict = dict()
    X_name = list_dir[-13]
    raw_X_ = pd.read_pickle(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta\%s'%X_name)
    date = X_name.split('_')[1].split('.')[0] #這天是指tuneta訓練完，再加1年的時間，EX : '2017-12-31'指的是訓練時間是2014-01-01~2016-12-31
    
    date_stamp = pd.to_datetime(date) - relativedelta(months=12)#扣N個月
    for mon in range(0,6):
        train_start = (date_stamp - relativedelta(months=12-mon)).strftime('%Y-%m-%d') #用12個月當訓練
        train_end = (date_stamp + relativedelta(months=0+mon)).strftime('%Y-%m-%d')
        backtest_end = date_stamp + relativedelta(months=2+mon) #避免在家資料的時候會少，所以多跑一個月，
        
        train_month = (backtest_end - relativedelta(months=1)).strftime('%Y-%m')
        backtest_end = backtest_end.strftime('%Y-%m-%d')
        print(train_start, train_month)
    
    
        ucb = UseCatBoost( train_start = train_start, train_end = train_end, backtest_end = backtest_end, 
                         target_col  = 'return', q_bins = 10)
        
        ucb.get_model(raw_X_, iterations = 10)
        y_new = ucb.get_predict(test = True)
        ucb.naive_plot( y_new, need = '3D')
        yNew_dict[train_month] = y_new
            


