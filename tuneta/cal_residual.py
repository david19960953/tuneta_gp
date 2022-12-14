# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:10:07 2022

@author: 仔仔
"""



import pandas as pd
import pickle
import numpy as np
import datetime
import os
from dateutil.relativedelta import relativedelta

with open(r'I:\我的雲端硬碟\colab_高曼_test\2011_to_2022_data.pickle', 'rb') as f:
    data_row = pickle.load(f)
    
   
data_row = data_row.loc['2012':,('Adj Close')]
col_list = [i for i in data_row.columns if i !='SPY']

col = 'VNT'

# resiAll_ll = []
for col in col_list[514:]:
    print(col)
    temp = data_row.loc[:,['SPY', col] ].dropna()
    if len(temp)<120:continue
    idd0 = temp.index[0]
    idd1 = temp.index[-1]   
    
    idd0_roll = temp.index[0]
    save_resi = []
    while True: 
        # print(idd0_roll)
        train_start = datetime.datetime(idd0_roll.year, idd0_roll.month, 1) #取那個月第一天
        train_end = train_start + relativedelta(months=6)
        backtest_end = train_end + relativedelta(months=1)
        
        
        if idd1 < backtest_end: break #執行別的
        try:
            條件1 = (temp.loc[(temp.index < train_end) & (temp.index >= train_start)].index)
            條件2 = (temp.loc[(temp.index < backtest_end) & (temp.index >= train_end)].index)
            
            temp_ = temp.loc[(temp.index < backtest_end) &  (temp.index >= train_start)]
            results1 = np.polyfit (temp_.loc[條件1, 'SPY'], temp_.loc[條件1, col] , 1)
            # Constant = results1[1]  
            predict = np.poly1d(results1)
            r =  temp_.loc[:,col] - predict(temp_['SPY'])   #找出訓練期間的殘差 
            roll_resi = ((r - r.rolling(20).mean().loc[條件2] )/r.rolling(20).std().loc[條件2] ).dropna()
            save_resi.append(roll_resi)
            idd0_roll += relativedelta(months=1)
        except Exception as e:
            print(e)
            break
    resiAll_ll.append(pd.concat(save_resi))
    
    
    
resiDiffAll_ll = []
for value in resiAll_ll:
    resiDiffAll_ll.append( value - value.ewm(alpha = 0.3).mean())
resiDiffAll_df =  pd.concat(resiDiffAll_ll, axis =1)
resiDiffAll_df = resiDiffAll_df.stack()
resiDiffAll_df.name = 'resi_diff'


resiAll_df = pd.concat(resiAll_ll, axis =1)
resiAll_df = resiAll_df.stack()
resiAll_df.name = 'residual'


residual =  pd.concat([resiAll_df,resiDiffAll_df], axis = 1)
# residual.to_pickle(r'I:\我的雲端硬碟\colab_高曼_test\residual.pickle')

# with open(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta\X_2021-06-30.pickle', 'rb') as f:
#     data_row = pickle.load(f)


# data_row = pd.concat([data_row,residual], axis = 1)
# aa = data_row.head(1000)








