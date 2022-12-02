# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:47:38 2022

@author: 仔仔
"""



from tuneta.tune_ta import TuneTA
import pandas as pd
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split
import yfinance as yf
import pickle
import alphalens
import missingno
import numpy as np
import pandas_ta as pta
import time
import os
# import matplotlib.pyplot as plt


from tuneta.transTuneTa import *
# from tuneta.transTuneTa import TransTa
# from tuneta.config import *


def get_top_volume(symbol_dict_adj , top_volume = 20):
    '''
    # top_volume : TYPE, int
    #     DESCRIPTION. The default is 20       
    #                  選擇要特定時間內成交量最大的
    '''
    
    
    # value = symbol_dict_adj['USTUSDT']
    tt_volume_df = pd.DataFrame()
    for symbol , value in symbol_dict_adj.items():
        if sum(pd.isna(value['Open'])) > 0: 
            print(sum(pd.isna(value['Open'])))
            continue
        if round(value['Open'].mean(),1)== 1:
            continue
        
        tt_volume = value['Quote asset volume'].sum()  #尋找最大交易量的標的
        tt_volume_df.loc[symbol , 'tt_volume'] = tt_volume
    
    tt_volume_df['tt_volume_rank'] = tt_volume_df['tt_volume'].rank(pct=True)
    tt_volume_df = tt_volume_df.sort_values(by = 'tt_volume' ,ascending = False)
    
    
    top_five_min_df = pd.DataFrame()
    for symbol , value in symbol_dict_adj.items():
        if symbol in tt_volume_df.index[:top_volume]:
            df = value['Open']
            df.name = symbol
            top_five_min_df = pd.concat([top_five_min_df , df] ,axis = 1)
    
    top_five_min_df.index = pd.to_datetime(top_five_min_df.index)
    missingno.matrix(top_five_min_df, freq='BQ')
    return top_five_min_df 


# symbol_df_adj = pd.read_csv(r'2011_to_2022_data.csv', index_col = 'Date',parse_dates = True)  #只有ticker 的close資料
# train_start = '2011-01-01'
# train_end = '2019-12-31'
# backtest_end = '2022-11-14'

# temp = symbol_df_adj[train_start : backtest_end]#.dropna(axis = 1)
# ticker_list = [i for i in temp.columns]
# data = yf.download(ticker_list,start = train_start,end = backtest_end)
# data.index = pd.to_datetime(data.index)

# with open(r'2011_to_2022_data.pickle', 'wb') as f:
#     pickle.dump(data, f)

path = r'H\\我的雲端硬碟\\colab_高曼_test\\'


with open(path + '2011_to_2022_data.pickle', 'rb') as f:
    data_row = pickle.load(f)


symbol_df_adj = pd.read_csv(path + '2011_to_2022_data.csv', index_col = 'Date',parse_dates = True)#看那個時候有甚麼標的






for year in range(2019,2022):
  print(year)
  train_start = '%s-01-01'%(year-3)
  train_end = '%s-12-31'%(year-1)
  backtest_end = '%s-12-31'%year



  temp = symbol_df_adj[train_start : backtest_end].dropna(axis = 1)
  ticker_list = [i for i in temp.columns]
  data = data_row.loc[train_start:backtest_end]

  
  volume_df = pd.DataFrame()
  for ticker in ticker_list:
      try:
          value = data.loc[:,(slice(None),ticker)]
          temp = value['Volume']
          temp.name = ticker
          volume_df = pd.concat([volume_df , temp] , axis = 1)
      except:pass


  volume_df.index = pd.to_datetime(volume_df.index)
  volume_df = volume_df.sort_index()
  volume_df_top = volume_df.loc[train_start : backtest_end ]
  volume_df_top50 = volume_df_top.loc[train_start : train_end ].sum() #計算交易量
  volume_df_top50 = volume_df_top50.sort_values(ascending = False)
  volume_df_top50.name = 'volume'

  top_number = 500  #看要選多少tick
  top_tikers = volume_df_top50.index[:top_number]


  data_ = data.loc[:, (slice(None),top_tikers)]
  data_ = data_.drop(columns = ('Close'))
  data_ = data_.rename(columns = {'Adj Close':'Close'})


  lag_period = 5 #目標是多少期後
  X = pd.DataFrame()
  for tiker in top_tikers:
      df = data_.loc[:, (slice(None),tiker)]
      df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
      df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
      df.columns = df.columns.str.lower() #統一標準
      df['sym'] = tiker
      df.set_index('sym', append=True, inplace=True)
      
      df['slope'] = pta.slope(df.close ,  length=lag_period).shift(-lag_period)  #往前移三個
      df['return'] = percent_return(df.close,length = lag_period ,offset= -lag_period)  #它內建已經移了
      # df['returnp'] = df.close.pct_change(lag_period).shift(-lag_period)  #等價於上  
      
      df = df.dropna(axis = 0)
      X = pd.concat([X, df], axis=0)
      # break
  X = X.sort_index()



  y = X['slope']
  # X = X.drop(columns=['return' , 'slope'])
  break
          

# import sys
# sys.exit()
saveTransta_ll = os.listdir(path +'saveTransta') #確認目錄內容，因為talib會載新的檔案，檔案順序好像會亂掉，所以要重新跑一次這個
transta_path = [i for i in saveTransta_ll if backtest_end in i][0]

with open(path + 'saveTransta\%s'%transta_path, 'rb') as f:
    transta = pickle.load(f)


main_list = list()
dd = transta.allFuncDict
for sym, v in X.groupby(level=1):
    transMul = trans_multi(v, dd)
    # output = transMul.transform()
    # print(sym)
    main_list.append(transMul)

def run_main(transMul):
    output = transMul.transform()
    return output


cpuCount = os.cpu_count()
from multiprocess import Pool    
start = time.time() # 開始測量執行時間
with Pool(processes= cpuCount ) as p:
    result = p.map(run_main, main_list )
end = time.time()
print( "執行時間為 %f 秒" % (end - start))
# feature = pd.concat(result, axis=0)


#資料前處理，把技術指標時間序列轉成輸入格式
result_select = []
ta_type_dict = trans_ta_type(ta_type)  
for see in result:
    seec = see.iloc[:].copy()
    seec = first_select(seec)
    seec = second_select(seec, ta_type_dict)
    result_select.append(seec)
    
feature = pd.concat(result_select, axis=0)    
feature = feature.sort_index(level = 0)
feature = feature.dropna()


cat_feature = third_select(seec,ta_type_dict)
# 設定catorgory, 不知道為啥只能用中文設
for col in cat_feature:
    df = feature[col]
    set_df = set(df)
    n_len = [str(i) for i in range(1,len(set_df)+1)]
    feature[col] = df.replace(set_df, n_len)


# 看tuneta訓練時間
times = [value['best_trial'].duration.total_seconds() for value in dd.values()]
inds = [value['function'].split("(")[0] for value in dd.values()]
df = pd.DataFrame(times, index = inds, columns =['second'])


X_ = pd.concat([feature ,X ],axis = 1, join = 'inner')

# with open(path + 'saveTransta\X_%s.pickle'%backtest_end, 'wb') as f:
#     pickle.dump(X_)

%matplotlib inline
import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns('FB')

# show sharpe ratio
qs.stats.sharpe(X_.open)

# or using extend_pandas() :)
stock.sharpe()
aa = X_.loc[(slice(None), 'SPY'),'close']
aa = aa.reset_index(level=[1],  drop=True)
qs.plots.snapshot(aa, title='Facebook Performance')
qs.reports.plots(aa, mode='full')




qs.reports.html(aa.pct_change(),aa, output = 'aaaa.html')


