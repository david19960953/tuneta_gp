# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 00:27:15 2022

@author: USER
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
import matplotlib.pyplot as plt


from tuneta.transTuneTa import *
from tuneta.transTuneTa import TransTa
from tuneta.config import *


with open('I:\我的雲端硬碟\colab_高曼_test\saveTransta/transta_2016-12-31.pickle', 'rb') as f:
    transta = pickle.load(f)

with open(r'I:\我的雲端硬碟\colab_高曼_test\2011_to_2022_data.pickle', 'rb') as f:
    data_row = pickle.load(f)

data_row = data_row.drop(columns = ('Close'))
data_row = data_row.rename(columns = {'Adj Close':'Close'})
data_row = data_row.stack()
data_row.columns = data_row.columns.str.lower() #統一標準

import os 
from dateutil.relativedelta import relativedelta
  
list_dir = os.listdir(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta')
list_dir = sorted([i for i in list_dir if 'X_' in i ])[:-1] #把2022 最後一個拿掉

yNew_dict = dict()
X_name = 'X_2016-12-31.pickle'
data_X = pd.read_pickle(r'I:\我的雲端硬碟\colab_高曼_test\saveTransta\%s'%X_name)

cor = data_X.corr()

func_group_df = []
for value in func_group.values():
    col = value[0]
    try:
        func_group_df.append(data_X.loc[:,col])
    except:pass




idd = transta.input_index
data_ = data_row.loc[idd]#.unstack()


main_list = list()
dd = transta.allFuncDict
for sym, v in data_.groupby(level=1):
    transMul = trans_multi(v, dd)
    # output = transMul.transform()
    # print(sym)
    main_list.append(transMul)
    # transMul = trans_multi(v, dd)
func_group = transMul.get_func_group()


def run_main(transMul):
    output = transMul.transform()
    return output
   
from multiprocess import Pool    
start = time.time() # 開始測量執行時間
with Pool(processes=12) as p:
    result = p.map(run_main, main_list )
end = time.time()
print( "執行時間為 %f 秒" % (end - start))



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




def prune( max_inter_correlation=0.7, top_prior=99999, top_post=99999):
        """
        Select most correlated with target, least intercorrelated
        :param top: Selects top x most correlated with target
        :param studies: From top x, keep y least intercorelated
        :return:
        """
        fitted = transta.allFuncDict

        fit_count = len(fitted)

        # Create feature correlation dataframe and remove duplicates
        feature_correlation = [
            [
                key,
                value['best_trial'].value,
            ]
            for key, value in fitted.items()
        ]
        feature_correlation = pd.DataFrame(feature_correlation).sort_values(
            by=1, ascending=False
        )
        feature_correlation = feature_correlation.drop_duplicates(
            subset=0, keep="first"
        )  # Duplicate indicators
        feature_correlation[1] = feature_correlation[1].round(4)
        feature_correlation = feature_correlation.drop_duplicates(
            subset=[1], keep="first"
        )  # Duplicate correlation

        # Filter top correlated features
        feature_correlation = feature_correlation.head(top_prior)
        self.fitted = [
            f for i, f in enumerate(self.fitted) if i in feature_correlation.index
        ]

        if not hasattr(self, "f_corr") or fit_count != len(self.fitted):
            self.features_corr()  
            #如果沒有self.f_corr，則執行 self.features_corr()，return self.f_corr

        # Iteratively removes least fit individual of most correlated pairs of studies
        # IOW, finds most correlated pairs, removes lest correlated to target until x studies
        components = list(range(len(self.fitted)))
        indices = list(range(len(self.fitted)))
        correlations = np.array(self.f_corr)

        most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
        correlation = correlations[most_correlated[0], most_correlated[1]]
        while correlation > max_inter_correlation:
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            correlation = correlations[most_correlated[0], most_correlated[1]]

        # Remove most correlated fits
        self.fitted = [self.fitted[i] for i in components][:top_post]

        # Recalculate correlation of fits
        self.target_corr()
        self.features_corr()
        
from tuneta.utils import col_name, distance_correlation
import itertools
from joblib import Parallel, delayed
def dc(p0,p1):
    df = pd.concat([p0, p1], axis=1).dropna()
    res = distance_correlation(
        np.array(df.iloc[:, 0]).astype(float), np.array(df.iloc[:, 1]).astype(float)
    )
    return res


import tuneta  
for p in pair_order_list:
    print(dc(p[0], p[1]))
    break

def features_corr(self):
    fns = []  # Function names
    cor = []  # Target Correlation
    features = []
    # for fit in func_group_df:
    #     fns.append(
    #         col_name(fit.function, fit.study.user_attrs["best_trial"].params)
    #     )
    #     cor.append(np.round(fit.study.user_attrs["best_trial"].value, 6))
    #     features.append(fit.study.user_attrs["best_trial"].user_attrs["res_y"])

    # # Feature must be same size for correlation and of type float
    # start = max([f.first_valid_index() for f in features])
    # features = [(f[f.index >= start]).astype(float) for f in features]

    # Inter Correlation
    func_group_df = [i.iloc[:,0] for i in func_group_df]
    pair_order_list = itertools.combinations(func_group_df, 2)
    correlations = []
    for p in pair_order_list:
        p
         correlations.append(dc(p[0], p[1]))
         
from multiprocess import Pool    
class run_dc():
    def __init__(self, p0,p1,distance_correlation):

        df = pd.concat([p0, p1], axis=1).dropna()
        self.p0 = np.array(df.iloc[:, 0]).astype(float)
        self.p1 = np.array(df.iloc[:, 1]).astype(float)
        self.distance_correlation = distance_correlation
    def dc(self):
        res = self.distance_correlation(
            self.p0,self.p1
        )
        return res
    
run_dc_ll = []
for p in pair_order_list:
    run_dc_ll.append(run_dc(p[0],p[1],distance_correlation))   
    # ss = run_dc(p[0],p[1],distance_correlation)
    # print(ss.dc())
    
def run_main(run_dc):
    res = run_dc.dc()
    return res

with Pool(processes=12) as p:
    result = p.map(run_main, run_dc_ll)
    

    correlations = Parallel(n_jobs=12)(
        delayed(dc)(p[0], p[1]) for p in pair_order_list
    )
    correlations = squareform(correlations)
    self.f_corr = pd.DataFrame(correlations, columns=fns, index=fns)












