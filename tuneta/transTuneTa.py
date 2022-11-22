# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:13:34 2022

@author: USER
"""

# from tuneta.tune_ta import TuneTA
import pandas as pd
import pickle
import re
import pandas_ta as pta
import time


class TransTa():
    def __init__(self, typename = 'ta'):
        self.allFuncDict = dict()
        self.typename = typename
        
        
    def input_tt(self,tt):
        
        '''
        tt = TuneTA(n_jobs = 1, verbose=True)
        tt.fit(X_train, y_train,
        indicators=['pta.macd', 'pta.rsi', 'tta.MACD'],
        ranges=[(2, 20)],
        trials=10,
        early_stop=5,)
        
        
        describe:
            tt is after fitting object of TuneTA
        
        '''
        fitted = tt.fitted
        self.input_index = fitted[0].study.user_attrs['best_trial'].user_attrs['res_y'].index
        self.start_time = self.input_index.levels[0][0]
        self.end_time = self.input_index.levels[0][1]
        
        
        if self.input_index.nlevels == 2:  # support 2 level inddex (data/symbol)
            self.start_time = self.input_index.levels[0][0]
            self.end_time = self.input_index.levels[0][-1]
            self.symbol_name = self.input_index.levels[1]
        else:
            self.start_time = self.input_index[0]
            self.end_time = self.input_index[-1]
        # allFeatures = pd.DataFrame()
        
        for optmize in fitted:
            function_ = optmize.function
            func_dict = optmize.study.user_attrs
            func_dict['function'] = function_
            
            del func_dict['best_trial'].user_attrs['res_y']  
            # features = self.transform( func_dict['best_trial'], X , func_dict['function'])
            # allFeatures = pd.concat([allFeatures , features] , axis = 1)
            self.allFuncDict[func_dict['name']] = func_dict
        print('done')


    def add_tt(self,tt):
        fitted = tt.fitted
        for optmize in fitted:
            function_ = optmize.function
            func_dict = optmize.study.user_attrs
            func_dict['function'] = function_

            
            del func_dict['best_trial'].user_attrs['res_y']  
            # features = self.transform( func_dict['best_trial'], X , func_dict['function'])
            # allFeatures = pd.concat([allFeatures , features] , axis = 1)
            self.allFuncDict[func_dict['name']] = func_dict
        
        


class trans_multi():
    def __init__(self, X, allFuncDict):
        self.X = X
        self.allFuncDict = allFuncDict

    def transform(self, columns=None ):
        """
        主程序
        Given X, create features of fitted studies
        :param X: Dataset with features used to create fitted studies
        :return:
        """
        # Remove trailing identifier in column list if present
        if columns is not None:
            columns = [re.sub(r"_[0-9]+$", "", s) for s in columns]

        self.X.columns = self.X.columns.str.lower()  # columns must be lower case
        # pool = ProcessPool(nodes=n_jobs)  # Number of jobs
        result = []

        # Iterate fitted studies and calculate TA with fitted parameter set
        self.time_df = pd.DataFrame()
        for function_ in self.allFuncDict.values():
            # Create field if no columns or is in columns list
            if columns is None or ind.res_y.name in columns:
               start = time.time()
               res = self.main( function_, )
               result.append(res)
               end = time.time()
               self.time_df.loc[function_['name'], 'time'] =  end - start
               # print(function_['name'], "執行時間為 %f 秒" % (end - start))
        # Blocking wait for asynchronous results
        # result = [res.get() for res in result]

        # Combine results into dataframe to return
        res = pd.concat(result, axis=1)
        res = pd.concat([res,self.X.close], axis=1)        
        return res
    
    
    def main(self,function_):
        res = self._trial( function_['best_trial'],  function_['function'] )
        return res
    
    
    def _trial(self, trial , function):

        """
        Calculate indicator using best fitted trial over X
        :param self:  Optuna study
        :param trial:  Optuna trial
        :param X:  dataset
        :return:
        """
        if self.X.index.nlevels == 2:  # support 2 level inddex (data/symbol)
            res = [
                self.trial_results(X, function, trial, sym=sym)
                for sym, X in self.X.groupby(level=1)
            ]
            res = pd.concat(res, axis=0).sort_index()
        else:
            res = self.trial_results(self.X, function, trial)
    
        # Create consistent column names with function string and params
        name = self.col_name(function, trial.params)
    
        # Append integer identifier to DF with multiple columns
        if len(res.columns) > 1:
            res.columns = [f"{name}_{i}" for i in range(len(res.columns))]
        else:
            res.columns = [f"{name}"]
        
        return res




    # Apply trial on multi index
    def trial_results(self,X, function, trial, sym=None):
        if sym:
            level_name = X.index.names[1]
            X = X.droplevel(1)
        try:
            res = eval(
                function
            )  # Eval contains reference to best trial (in argument) to re-use original parameters
        except Exception as e:
            raise Exception(e)
        if isinstance(res, tuple):
            try:
                res = pd.DataFrame(res).T
            except Exception as e:
                print("Error:")
                print(f"Function: {function}")
                print(f"X Length:  {len(X)}")
                for k, v in enumerate(res):
                    u, c = np.unique(v.index, return_counts=True)
                    dup = u[c > 1]
                    print(f"Series {k} duplicates: {len(dup)}")
                    print(v)
                raise Exception(e)
    
        res = pd.DataFrame(res, index=X.index)  # Ensure result aligns with X
        if sym:
            res[level_name] = sym
            res.set_index(level_name, append=True, inplace=True)
        return res




    def col_name(self, function, study_best_params):
        """
        Create consistent column names given string function and params
        :param function:  Function represented as string
        :param study_best_params:  Params for function
        :return:
        """
    
        # Optuna string of indicator
        function_name = function.split("(")[0].replace(".", "_")
    
        # Optuna string of parameters
        params = (
            re.sub("[^0-9a-zA-Z_:,]", "", str(study_best_params))
            .replace(",", "_")
            .replace(":", "_")
        )
    
        # Concatenate name and params to define
        col = f"{function_name}_{params}"
    
        # Remove any trailing underscores
        col = re.sub(r"_$", "", col)
        return col








ta_type ={
  'pta_atr_length_5_mamode_dema': 'ts',
  'pta_pdist': 'ts',
  'pta_true_range': 'ts',
  'pta_psar_0': 'tsma',
  'pta_psar_1': 'tsma',
  'pta_psar_2': 'station',
  'pta_psar_3': 'station',
  'pta_rma_length_3': 'tsma',
  'pta_aberration_length_3_atr_length_5_0': 'tsma',
  'pta_aberration_length_3_atr_length_5_1': 'tsma',
  'pta_aberration_length_3_atr_length_5_2': 'tsma',
  'pta_aberration_length_3_atr_length_5_3': 'station',
  'pta_midpoint_length_3': 'tsma',
  'pta_pwma_length_3': 'tsma',
  'pta_swma_length_3': 'tsma',
  'pta_trima_length_3': 'tsma',
  'pta_sinwma_length_3': 'tsma',
  'pta_linreg_length_3': 'tsma',
  'pta_sma_length_3': 'tsma',
  'pta_ema_length_3': 'tsma',
  'pta_midprice_length_3': 'tsma',
  'pta_fwma_length_3': 'tsma',
  'pta_jma_length_3': 'tsma',
  'pta_median_length_3': 'tsma',
  'pta_quantile_length_3_q_5': 'tsma',
  'pta_wma_length_3': 'tsma',
  'pta_zlma_length_3': 'tsma',
  'pta_decay_length_3': 'tsma',
  'pta_hilo_high_length_3_low_length_3_mamode_wma_0': 'tsma',
  'pta_hilo_high_length_3_low_length_3_mamode_wma_1': 'tsma',
  'pta_hilo_high_length_3_low_length_3_mamode_wma_2': 'tsma',
  'pta_dema_length_3': 'tsma',
  'pta_bbands_length_3_ddof_3_mamode_wma_0': 'tsma',
  'pta_bbands_length_3_ddof_3_mamode_wma_1': 'tsma',
  'pta_bbands_length_3_ddof_3_mamode_wma_2': 'tsma',
  'pta_bbands_length_3_ddof_3_mamode_wma_3': 'station',
  'pta_bbands_length_3_ddof_3_mamode_wma_4': 'station',
  'pta_tema_length_3': 'tsma',
  'pta_hl2': 'tsma',
  'pta_ohlc4': 'tsma',
  'pta_mcgd_length_3': 'tsma',
  'pta_vwap': 'tsma',
  'pta_hlc3': 'tsma',
  'pta_wcp': 'tsma',
  'pta_supertrend_length_3_0': 'tsma',
  'pta_supertrend_length_3_1': 'binary',
  'pta_supertrend_length_3_2': 'tsma',
  'pta_supertrend_length_3_3': 'tsma',
  'pta_donchian_lower_length_3_upper_length_5_0': 'tsma',
  'pta_donchian_lower_length_3_upper_length_5_1': 'tsma',
  'pta_donchian_lower_length_3_upper_length_5_2': 'tsma',
  'pta_ssf_length_3': 'tsma',
  'pta_hwma': 'tsma',
  'pta_hwc_0': 'tsma',
  'pta_hwc_1': 'tsma',
  'pta_hwc_2': 'tsma',
  'pta_alma_length_3': 'tsma',
  'pta_cksp_p_3_q_5_0': 'tsma',
  'pta_cksp_p_3_q_5_1': 'tsma',
  'pta_t3_length_3': 'tsma',
  'pta_kama_length_4_fast_2_slow_2': 'tsma',
  'pta_vidya_length_3': 'tsma',
  'pta_kc_length_5_mamode_dema_0': 'tsma',
  'pta_kc_length_5_mamode_dema_1': 'tsma',
  'pta_kc_length_5_mamode_dema_2': 'tsma',
  'pta_accbands_length_5_mamode_dema_0': 'tsma',
  'pta_accbands_length_5_mamode_dema_1': 'tsma',
  'pta_accbands_length_5_mamode_dema_2': 'tsma',
  'pta_mad_length_3': 'ts',
  'pta_stdev_length_3_ddof_5': 'ts',
  'pta_eri_length_3_0': 'ts',
  'pta_eri_length_3_1': 'ts',
  'pta_thermo_length_4_long_3_short_2_mamode_swma_0': 'ts',
  'pta_thermo_length_4_long_3_short_2_mamode_swma_1': 'ts',
  'pta_thermo_length_4_long_3_short_2_mamode_swma_2': 'binary',
  'pta_thermo_length_4_long_3_short_2_mamode_swma_3': 'binary',
  'pta_dm_length_5_mamode_dema_0': 'ts',
  'pta_dm_length_5_mamode_dema_1': 'ts',
  'pta_variance_length_3_ddof_5': 'ts',
  'pta_macd_fast_4_slow_2_signal_2_0': 'ts',
  'pta_macd_fast_4_slow_2_signal_2_1': 'ts',
  'pta_macd_fast_4_slow_2_signal_2_2': 'ts',
  'pta_dpo_length_3': 'ts',
  'pta_qstick_length_3': 'ts',
  'pta_mom_length_3': 'ts',
  'pta_slope_length_3': 'ts',
  'pta_apo_fast_3_slow_5_mamode_swma': 'ts',
  'pta_ao_fast_3_slow_5': 'ts',
  'pta_squeeze_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_0': 'ts',
  'pta_squeeze_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_1': 'binary',
  'pta_squeeze_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_2': 'binary',
  'pta_squeeze_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_3': 'binary',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_0': 'ts',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_1': 'binary',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_2': 'binary',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_3': 'binary',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_4': 'binary',
  'pta_squeeze_pro_bb_length_3_kc_length_3_mom_length_5_mom_smooth_3_mamode_pwma_5': 'binary',
  'pta_eom_length_3': 'ts',
  'pta_pvol': 'ts',
  'pta_efi_length_5_mamode_dema': 'station',
  'pta_pvi_length_3': 'station',
  'pta_ad': 'ts',
  'pta_nvi_length_3': 'station',
  'pta_kvo_fast_4_slow_3_signal_2_mamode_swma_0': 'ts',
  'pta_kvo_fast_4_slow_3_signal_2_mamode_swma_1': 'ts',
  'pta_pvt': 'ts',
  'pta_adosc_fast_3_slow_5': 'ts',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_0': 'station',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_1': 'station',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_2': 'station',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_3': 'station',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_4': 'station',  
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_5': 'binary',
  'pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_6': 'binary',
  'pta_obv': 'ts',
  'pta_natr_length_5_mamode_dema': 'station',
  'pta_kdj_length_3_signal_5_0': 'station',
  'pta_kdj_length_3_signal_5_1': 'station',
  'pta_kdj_length_3_signal_5_2': 'station',
  'pta_fisher_length_3_signal_5_0': 'station',
  'pta_fisher_length_3_signal_5_1': 'station',
  'pta_massi_fast_3_slow_5': 'station',
  'pta_qqe_length_3_smooth_5_0': 'station',
  'pta_qqe_length_3_smooth_5_1': 'station',
  'pta_qqe_length_3_smooth_5_2': 'station',
  'pta_qqe_length_3_smooth_5_3': 'station',
  'pta_pvo_fast_4_slow_2_signal_2_0': 'station',
  'pta_pvo_fast_4_slow_2_signal_2_1': 'station',
  'pta_pvo_fast_4_slow_2_signal_2_2': 'station',
  'pta_stoch_k_4_d_3_smooth_k_2_mamode_swma_0': 'station',
  'pta_stoch_k_4_d_3_smooth_k_2_mamode_swma_1': 'station',
  'pta_chop_length_3_atr_length_5': 'station',
  'pta_smi_fast_4_slow_2_signal_2_0': 'station',
  'pta_smi_fast_4_slow_2_signal_2_1': 'station',
  'pta_smi_fast_4_slow_2_signal_2_2': 'station',
  'pta_tsi_fast_4_slow_3_signal_2_mamode_swma_0': 'station',
  'pta_tsi_fast_4_slow_3_signal_2_mamode_swma_1': 'station',
  'pta_rsi_length_3': 'station',
  'pta_cmo_length_3': 'station',
  'pta_ui_length_3': 'station',
  'pta_uo_fast_4_medium_2_slow_2': 'station',
  'pta_pgo_length_3': 'station',
  'pta_willr_length_3': 'station',
  'pta_rsx_length_3': 'station',
  'pta_trix_length_3_signal_5_0': 'station',
  'pta_trix_length_3_signal_5_1': 'station',
  'pta_vortex_length_3_0': 'station',
  'pta_vortex_length_3_1': 'station',
  'pta_roc_length_3': 'station',
  'pta_coppock_length_4_fast_2_slow_2': 'station',
  'pta_entropy_length_3': 'station',
  'pta_kst_roc1_3_roc2_2_roc3_4_roc4_3_sma1_2_sma2_3_sma3_2_sma4_5_signal_3_0': 'station',
  'pta_kst_roc1_3_roc2_2_roc3_4_roc4_3_sma1_2_sma2_3_sma3_2_sma4_5_signal_3_1': 'station',
  'pta_cmf_length_3': 'station',
  'pta_cg_length_3': 'station',
  'pta_stc_tclength_3_fast_5_slow_4_0': 'station',
  'pta_stc_tclength_3_fast_5_slow_4_1': 'station',
  'pta_stc_tclength_3_fast_5_slow_4_2': 'station',
  'pta_ppo_fast_4_slow_3_signal_2_mamode_swma_0': 'station',
  'pta_ppo_fast_4_slow_3_signal_2_mamode_swma_1': 'station',
  'pta_ppo_fast_4_slow_3_signal_2_mamode_swma_2': 'station',
  'pta_rvgi_length_3_swma_length_5_0': 'station',
  'pta_rvgi_length_3_swma_length_5_1': 'station',
  'pta_bias_length_5_mamode_dema': 'station',
  'pta_mfi_length_3': 'station',
  'pta_rvi_length_3': 'station',
  'pta_cci_length_3': 'station',
  'pta_aroon_length_3_0': 'four',
  'pta_aroon_length_3_1': 'four',
  'pta_aroon_length_3_2': 'seven',
  'pta_ttm_trend_length_3': 'binary',
  'pta_cti_length_3': 'station',
  'pta_decreasing_length_3': 'binary',
  'pta_cfo_length_3': 'station',
  'pta_zscore_length_3': 'station',
  'pta_stochrsi_length_3_rsi_length_3_k_5_d_3_mamode_pwma_0': 'station',
  'pta_stochrsi_length_3_rsi_length_3_k_5_d_3_mamode_pwma_1': 'station',
  'pta_inertia_length_3_rvi_length_5': 'station',
  'pta_bop': 'station',
  'pta_increasing_length_3': 'binary',
  'pta_adx_length_3_lensig_3_mamode_wma_0': 'station',
  'pta_adx_length_3_lensig_3_mamode_wma_1': 'station',
  'pta_adx_length_3_lensig_3_mamode_wma_2': 'station',
  'pta_er_length_3': 'station',
  'pta_kurtosis_length_5': 'station',
  'pta_psl_length_3': 'four',
  'pta_vhf_length_3': 'station',
  'pta_amat_fast_4_slow_3_lookback_2_mamode_swma_0': 'binary',
  'pta_amat_fast_4_slow_3_lookback_2_mamode_swma_1': 'binary',
  'pta_skew_length_4': 'station',
  'pta_brar_length_3_0': 'station',
  'pta_brar_length_3_1': 'station',
  'pta_vp': 'ts',
  'pta_psar_m': 'ts',               #合併後的, m字尾的代表有修改過的
  'pta_supertrend_m': 'tsma',
  'pta_qqe_m':'station'
  }




'''
ts =[5,6,7,73,74,75,76,77,78,
 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,97,103,
104,107,109, 110, 111, 112, 113, 114, 115,116,119]


tsma = [8,9,12,13,14,15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
42, 43, 44, 45, 46, 47, 48, 49, 51, 52,53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
63, 64, 65, 66, 67, 68, 69, 70, 71,72]


binary = [50,79,80,94, 95, 96, 98, 99, 100, 101,102,117,118,175,177,184,192,193]
four = [172,173,190]
seven = [174]

station = 剩下



代號說明
ts: 代表他們沒有固定上下值域，且跟價格不太像(應該跟volume有關)，不知是不是定態
tsma: 代表他們是時間序列且跟ohlc有關，是某種MA系列的轉換，所以跟價格很相似
binary、four、seven: 二,四,七元分類
station: 有固定上下界，可以一起比較，但不一定是定態
'''


'''
first_select說明

8	pta_psar_0 = 'ts'
9	pta_psar_1 = 'ts'
8+9要何在一起，他們是互補的，不然會有na


34	pta_hilo_high_length_3_low_length_3_mamode_wma_1 = 'tsma'
35	pta_hilo_high_length_3_low_length_3_mamode_wma_2 = 'tsma'
不知道為啥有na，可以刪了

51	pta_supertrend_length_3_2 = 'tsma'
52	pta_supertrend_length_3_3 = 'tsma'
51+52 要何在一起，他們是互補的，不然會有na


117	pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_4 = 'binary'  #(1,0)
118	pta_aobv_fast_3_slow_3_max_lookback_5_min_lookback_3_mamode_pwma_5 = 'binary'  #(1,0)
他們是互補的，只要選一個就行


129	pta_qqe_length_3_smooth_5_2 = 'station'
130	pta_qqe_length_3_smooth_5_3 = 'station'
他們是互補的，要合併再一起，中間會有NA


192	pta_amat_fast_4_slow_3_lookback_2_mamode_swma_0 = 'binary' #(1,0)
193	pta_amat_fast_4_slow_3_lookback_2_mamode_swma_1 = 'binary' #(1,0)
他們是互補的，只要選一個就行

# 195	pta_brar_length_3_0 = 'station'
# 196	pta_brar_length_3_1 = 'station'
# 他們數字很接近，也許可以只選一個就好


##新增錯誤
'pta_adx_length_3_lensig_3_mamode_wma_0': 'station',
後面是NA

pta_ichimoku_tenkan_4_kijun_2_senkou_2_0
pta_ichimoku_tenkan_4_kijun_2_senkou_2_1
這邊有2個全NA的要刪掉


'''

#  把ta_type的字典格式轉成需要的格式
def trans_ta_type(ta_type):
    ta_type_df = pd.DataFrame(ta_type.values() ,index = ta_type.keys() , columns = ['type'])
    ta_type_df.reset_index(inplace =True)
    ta_type_df.rename(columns = {'index': 'col'}, inplace =True)
    ta_type_df['indicator'] = ta_type_df['col'].apply(lambda x: 'pta_'+x.split('_')[1])
    ta_type_dict = dict()
    for key , value in ta_type_df.groupby('indicator'):
        if len(value) == 1:
            value.index = ['0']
        else:
            value.index = value['col'].apply(lambda x: str(x[-1]))
        ta_type_dict[key] = value
    return ta_type_dict




#把有問題的資料作整理以及合併
def first_select(see): 
    seec = see.iloc[:].copy()
    
    trans_df = pd.DataFrame()
    col_ll = seec.columns
    # first_step 
    pta_psar = 'pta_psar'
    pta_psar_ll = [col for col in col_ll if pta_psar in col]
    if len(pta_psar_ll) != 0:
        temp1 = seec[pta_psar_ll[0]].dropna()
        temp2 = seec[pta_psar_ll[1]].dropna()
        temp_all = pd.concat([temp1,temp2]).sort_index()
        temp_all.name = 'pta_psar_m'
        trans_df = pd.concat([trans_df, temp_all], axis = 1)
        seec = seec.drop(columns =[pta_psar_ll[0], pta_psar_ll[1], ] )

    # second_step 
    pta_hilo = 'pta_hilo'
    pta_hilo_ll = [col for col in col_ll if pta_hilo in col]
    if len(pta_hilo_ll) != 0:
        seec = seec.drop(columns =[pta_hilo_ll[1], pta_hilo_ll[2], ] )   

    #third step
    pta_supertrend = 'pta_supertrend'
    pta_supertrend_ll = [col for col in col_ll if pta_supertrend in col]
    if len(pta_supertrend_ll) != 0:
        temp1 = seec[pta_supertrend_ll[2]].dropna()
        temp2 = seec[pta_supertrend_ll[3]].dropna()
        temp_all = pd.concat([temp1,temp2]).sort_index()
        temp_all.name = 'pta_supertrend_m'
        trans_df = pd.concat([trans_df, temp_all], axis = 1)
        seec = seec.drop(columns =[pta_supertrend_ll[2], pta_supertrend_ll[3], ] )

    #fourth step
    pta_aobv = 'pta_aobv'
    pta_aobv_ll = [col for col in col_ll if pta_aobv in col]
    if len(pta_aobv_ll) != 0:   
        seec = seec.drop(columns =[pta_aobv_ll[5] ] )  #刪掉5

    #fiveth step
    pta_qqe = 'pta_qqe'
    pta_qqe_ll = [col for col in col_ll if pta_qqe in col]
    if len(pta_qqe_ll) != 0:
        temp1 = seec[pta_qqe_ll[2]].dropna()
        temp2 = seec[pta_qqe_ll[3]].dropna()
        temp_all = pd.concat([temp1,temp2]).sort_index()
        temp_all.name = 'pta_qqe_m'
        trans_df = pd.concat([trans_df, temp_all], axis = 1)
        seec = seec.drop(columns =[pta_qqe_ll[2], pta_qqe_ll[3], ] )

    #sixth step
    pta_amat = 'pta_amat'
    pta_amat_ll = [col for col in col_ll if pta_amat in col]
    if len(pta_amat_ll) != 0:      
        seec = seec.drop(columns =[pta_amat_ll[1] ] )  #刪掉1
    
    #seventh step  
    pta_adx = 'pta_adx'
    pta_adx_ll = [col for col in col_ll if pta_adx in col]
    if len(pta_adx_ll) != 0:    
        seec = seec.drop(columns =[pta_adx_ll[0] ] )  #刪掉0
    
    pta_ichimoku_tenkan = 'pta_ichimoku_tenkan'
    pta_ichimoku_tenkan_ll = [col for col in col_ll if pta_ichimoku_tenkan in col]
    if len(pta_ichimoku_tenkan_ll) != 0:
        seec = seec.drop(columns =pta_ichimoku_tenkan_ll )  #刪掉0   
    # 合併
    seec = pd.concat([seec, trans_df],axis = 1)
    return seec




# for col in col_ll :
#     if key == 'pta_'+col.split('_')[1]:
#         print(key)




def second_select(see,ta_type_dict):
    seec =see.iloc[:].copy()
    seec = seec.fillna(method = 'pad', axis = 0)
    seec = seec.dropna()
    close = seec['close']
    seec = seec.drop(columns = 'close')
    col_ll = seec.columns
    
    for key in ta_type_dict.keys():
        if key == 'pta_squeeze':  #他比較特別又有子集bb 跟pro_bb
            key_ll = [col for col in col_ll if key == 'pta_'+col.split('_')[1]]
            value = ta_type_dict[key]
            key_ll_bb = key_ll[:4]
            key_ll_pro = key_ll[4:]
            value_bb = value.iloc[:4, :]
            value_pro = value.iloc[4:, :]  
            
            for col in key_ll_bb:
                idd  = col[-1]
                if value_bb.loc[idd, 'type'] == 'tsma' :
                    seec[col+'_close'] = seec[col]/close  
                    seec[col] = seec[col].pct_change()     
                elif value_bb.loc[idd, 'type'] == 'ts':
                    seec[col] = seec[col]/close
                    
            for col in key_ll_pro:
                idd  = col[-1]
                if value_pro.loc[idd, 'type'] == 'tsma':
                    seec[col+'_close'] = seec[col]/close  
                    seec[col] = seec[col].pct_change()   
                elif value_pro.loc[idd, 'type'] == 'ts':
                    seec[col] = seec[col]/close
                    
            continue
        
        key_ll = [col for col in col_ll if key == 'pta_'+col.split('_')[1]]
        if len(key_ll) ==0:continue
        # print(key)
        if len(key_ll) ==1: #如果只有一種
            value = ta_type_dict[key]
            col = key_ll[0]
            if value['type'].values[0] == 'tsma' :
                seec[col+'_close'] = seec[col]/close  
                seec[col] = seec[col].pct_change()
                continue
            elif value['type'].values[0] == 'ts':
                seec[col] = seec[col]/close
                continue
            else:continue
            
        value = ta_type_dict[key]         
        for col in key_ll: #如果有多種
            idd  = col[-1]
            if value.loc[idd, 'type'] == 'tsma':
                seec[col+'_close'] = seec[col]/close  
                seec[col] = seec[col].pct_change()       
            elif value.loc[idd, 'type'] == 'ts':
                seec[col] = seec[col]/close
                
    return seec




# 找那些column是cat_feature
def third_select(see,ta_type_dict):
    seec =see.iloc[:].copy()
    seec = seec.fillna(method = 'pad', axis = 0)
    seec = seec.dropna()
    col_ll = seec.columns
    
    cat_feature = []
    for key in ta_type_dict.keys():
        if key == 'pta_squeeze':  #他比較特別又有子集bb 跟pro_bb
            key_ll = [col for col in col_ll if key == 'pta_'+col.split('_')[1]]
            value = ta_type_dict[key]
            key_ll_bb = key_ll[:4]
            key_ll_pro = key_ll[4:]
            value_bb = value.iloc[:4, :]
            value_pro = value.iloc[4:, :]  
            
            for col in key_ll_bb:
                idd  = col[-1]           
                if value_bb.loc[idd,'type'] == 'binary' or value_bb.loc[idd,'type'] == 'four' or value_bb.loc[idd,'type'] == 'seven':
                    cat_feature.append(col) 
                    
            for col in key_ll_pro:
                idd  = col[-1]
                if value_pro.loc[idd,'type'] == 'binary' or value_pro.loc[idd,'type'] == 'four' or value_pro.loc[idd,'type'] == 'seven':
                    cat_feature.append(col)     
            continue
        
        key_ll = [col for col in col_ll if key == 'pta_'+col.split('_')[1]]
        key_ll = [col for col in key_ll if   col.split('_')[-1] != 'close']
        if len(key_ll) ==0:continue
        # print(key)
        if len(key_ll) ==1:
            value = ta_type_dict[key]
            col = key_ll[0]
            if value['type'].values[0] == 'binary' or value['type'].values[0] == 'four' or value['type'].values[0] == 'seven':
                cat_feature.append(col)
                continue
            else:continue
            
            
        value = ta_type_dict[key]        
        for col in key_ll:
            idd  = col[-1]
            if value.loc[idd,'type'] == 'binary' or value.loc[idd,'type'] == 'four' or value.loc[idd,'type'] == 'seven':
                cat_feature.append(col)    
            else:continue                
    return cat_feature
    

'''

360期，100個標的，訓練一次需要的時間(單純只訓練一次，不加 k_mens)

參數
# tt.fit(X_train, y_train,
#     indicators=['pta'],
#     ranges=[(2, 5)],
#     trials=5,
#     early_stop=3,)
'''


# 575.99
tuneta_time = {
  0: {'Indicator': 'pta.amat', 'Times': 70.42, 'trial1': False},
  1: {'Indicator': 'pta.increasing', 'Times': 67.43, 'trial1': False},
  2: {'Indicator': 'pta.ttm_trend', 'Times': 67.33, 'trial1': False},
  3: {'Indicator': 'pta.decreasing', 'Times': 66.68, 'trial1': False},
  4: {'Indicator': 'pta.mcgd', 'Times': 19.75, 'trial1': False},
  5: {'Indicator': 'pta.vidya', 'Times': 11.54, 'trial1': False},
  6: {'Indicator': 'pta.adx', 'Times': 9.97, 'trial1': False},
  7: {'Indicator': 'pta.ssf', 'Times': 9.39, 'trial1': False},
  8: {'Indicator': 'pta.squeeze_pro', 'Times': 9.02, 'trial1': False},
  9: {'Indicator': 'pta.alma', 'Times': 8.94, 'trial1': False},
  10: {'Indicator': 'pta.thermo', 'Times': 6.6, 'trial1': False},
  11: {'Indicator': 'pta.jma', 'Times': 6.57, 'trial1': False},
  12: {'Indicator': 'pta.inertia', 'Times': 5.21, 'trial1': False},
  13: {'Indicator': 'pta.kama', 'Times': 4.97, 'trial1': False},
  14: {'Indicator': 'pta.cti', 'Times': 4.83, 'trial1': False},
  15: {'Indicator': 'pta.kvo', 'Times': 4.83, 'trial1': False},
  16: {'Indicator': 'pta.stoch', 'Times': 4.77, 'trial1': False},
  17: {'Indicator': 'pta.mad', 'Times': 4.51, 'trial1': False},
  18: {'Indicator': 'pta.rvgi', 'Times': 4.45, 'trial1': False},
  19: {'Indicator': 'pta.sinwma', 'Times': 4.26, 'trial1': False},
  20: {'Indicator': 'pta.linreg', 'Times': 4.1, 'trial1': False},
  21: {'Indicator': 'pta.fisher', 'Times': 3.98, 'trial1': False},
  22: {'Indicator': 'pta.stochrsi', 'Times': 3.91, 'trial1': False},
  23: {'Indicator': 'pta.cfo', 'Times': 3.81, 'trial1': False},
  24: {'Indicator': 'pta.pvi', 'Times': 3.74, 'trial1': False},
  25: {'Indicator': 'pta.rsx', 'Times': 3.54, 'trial1': False},
  26: {'Indicator': 'pta.rvi', 'Times': 3.53, 'trial1': False},
  27: {'Indicator': 'pta.squeeze', 'Times': 3.51, 'trial1': False},
  28: {'Indicator': 'pta.aobv', 'Times': 3.32, 'trial1': False},
  29: {'Indicator': 'pta.brar', 'Times': 3.3, 'trial1': False},
  30: {'Indicator': 'pta.kst', 'Times': 3.27, 'trial1': False},
  31: {'Indicator': 'pta.ui', 'Times': 3.13, 'trial1': False},
  32: {'Indicator': 'pta.decay', 'Times': 3.12, 'trial1': False},
  33: {'Indicator': 'pta.qstick', 'Times': 3.11, 'trial1': False},
  34: {'Indicator': 'pta.vhf', 'Times': 3.09, 'trial1': False},
  35: {'Indicator': 'pta.kdj', 'Times': 2.99, 'trial1': False},
  36: {'Indicator': 'pta.massi', 'Times': 2.98, 'trial1': False},
  37: {'Indicator': 'pta.eom', 'Times': 2.83, 'trial1': False},
  38: {'Indicator': 'pta.cg', 'Times': 2.82, 'trial1': False},
  39: {'Indicator': 'pta.smi', 'Times': 2.72, 'trial1': False},
  40: {'Indicator': 'pta.cmf', 'Times': 2.68, 'trial1': False},
  41: {'Indicator': 'pta.pwma', 'Times': 2.67, 'trial1': False},
  42: {'Indicator': 'pta.fwma', 'Times': 2.67, 'trial1': False},
  43: {'Indicator': 'pta.trix', 'Times': 2.61, 'trial1': False},
  44: {'Indicator': 'pta.macd', 'Times': 2.57, 'trial1': False},
  45: {'Indicator': 'pta.swma', 'Times': 2.55, 'trial1': False},
  46: {'Indicator': 'pta.vortex', 'Times': 2.53, 'trial1': False},
  47: {'Indicator': 'pta.chop', 'Times': 2.53, 'trial1': False},
  48: {'Indicator': 'pta.tsi', 'Times': 2.52, 'trial1': False},
  49: {'Indicator': 'pta.pgo', 'Times': 2.52, 'trial1': False},
  50: {'Indicator': 'pta.tema', 'Times': 2.5, 'trial1': False},
  51: {'Indicator': 'pta.efi', 'Times': 2.49, 'trial1': False},
  52: {'Indicator': 'pta.variance', 'Times': 2.48, 'trial1': False},
  53: {'Indicator': 'pta.psl', 'Times': 2.42, 'trial1': False},
  54: {'Indicator': 'pta.eri', 'Times': 2.42, 'trial1': False},
  55: {'Indicator': 'pta.dpo', 'Times': 2.41, 'trial1': False},
  56: {'Indicator': 'pta.dema', 'Times': 2.39, 'trial1': False},
  57: {'Indicator': 'pta.sma', 'Times': 2.39, 'trial1': False},
  58: {'Indicator': 'pta.zscore', 'Times': 2.34, 'trial1': False},
  59: {'Indicator': 'pta.entropy', 'Times': 2.33, 'trial1': False},
  60: {'Indicator': 'pta.stdev', 'Times': 2.33, 'trial1': False},
  61: {'Indicator': 'pta.coppock', 'Times': 2.3, 'trial1': False},
  62: {'Indicator': 'pta.pvo', 'Times': 2.28, 'trial1': False},
  63: {'Indicator': 'pta.midpoint', 'Times': 2.27, 'trial1': False},
  64: {'Indicator': 'pta.quantile', 'Times': 2.26, 'trial1': False},
  65: {'Indicator': 'pta.zlma', 'Times': 2.24, 'trial1': False},
  66: {'Indicator': 'pta.er', 'Times': 2.22, 'trial1': False},
  67: {'Indicator': 'pta.adosc', 'Times': 2.21, 'trial1': False},
  68: {'Indicator': 'pta.rma', 'Times': 2.18, 'trial1': False},
  69: {'Indicator': 'pta.slope', 'Times': 2.18, 'trial1': False},
  70: {'Indicator': 'pta.cmo', 'Times': 2.13, 'trial1': False},
  71: {'Indicator': 'pta.t3', 'Times': 2.13, 'trial1': False},
  72: {'Indicator': 'pta.mfi', 'Times': 2.11, 'trial1': False},
  73: {'Indicator': 'pta.midprice', 'Times': 2.11, 'trial1': False},
  74: {'Indicator': 'pta.skew', 'Times': 2.09, 'trial1': False},
  75: {'Indicator': 'pta.ao', 'Times': 2.05, 'trial1': False},
  76: {'Indicator': 'pta.trima', 'Times': 2.05, 'trial1': False},
  77: {'Indicator': 'pta.wma', 'Times': 2.02, 'trial1': False},
  78: {'Indicator': 'pta.ema', 'Times': 2.02, 'trial1': False},
  79: {'Indicator': 'pta.cci', 'Times': 2.01, 'trial1': False},
  80: {'Indicator': 'pta.bias', 'Times': 2.01, 'trial1': False},
  81: {'Indicator': 'pta.atr', 'Times': 2.01, 'trial1': False},
  82: {'Indicator': 'pta.natr', 'Times': 1.99, 'trial1': False},
  83: {'Indicator': 'pta.uo', 'Times': 1.99, 'trial1': False},
  84: {'Indicator': 'pta.ppo', 'Times': 1.98, 'trial1': False},
  85: {'Indicator': 'pta.median', 'Times': 1.96, 'trial1': False},
  86: {'Indicator': 'pta.roc', 'Times': 1.96, 'trial1': False},
  87: {'Indicator': 'pta.willr', 'Times': 1.91, 'trial1': False},
  88: {'Indicator': 'pta.apo', 'Times': 1.86, 'trial1': False},
  89: {'Indicator': 'pta.dm', 'Times': 1.86, 'trial1': False},
  90: {'Indicator': 'pta.rsi', 'Times': 1.82, 'trial1': False},
  91: {'Indicator': 'pta.vwap', 'Times': 0.58, 'trial1': True},
  92: {'Indicator': 'pta.hwma', 'Times': 0.54, 'trial1': True},
  93: {'Indicator': 'pta.ohlc4', 'Times': 0.36, 'trial1': True},
  94: {'Indicator': 'pta.hl2', 'Times': 0.32, 'trial1': True},
  95: {'Indicator': 'pta.bop', 'Times': 0.32, 'trial1': True},
  96: {'Indicator': 'pta.true_range', 'Times': 0.3, 'trial1': True},
  97: {'Indicator': 'pta.pvt', 'Times': 0.29, 'trial1': True},
  98: {'Indicator': 'pta.ad', 'Times': 0.23, 'trial1': True},
  99: {'Indicator': 'pta.pvol', 'Times': 0.22, 'trial1': True},
  100: {'Indicator': 'pta.hlc3', 'Times': 0.64, 'trial1': True},
  101: {'Indicator': 'pta.wcp', 'Times': 0.5, 'trial1': True},
  102: {'Indicator': 'pta.pdist', 'Times': 0.58, 'trial1': True},
  103: {'Indicator': 'pta.obv', 'Times': 0.62, 'trial1': True}}















if __name__ == '__main__':
    import os
    import time
    from sklearn.model_selection import train_test_split
    # from tuneta.tune_ta import TuneTA
    import pandas as pd
    import pickle
    
    with open(r'need_data.pickle', 'rb') as f:
        need_data = pickle.load( f)
    X = need_data['X']
    del need_data
    y = X['slope']
    X = X.drop(columns=['return' , 'slope'])
    
    with open(r'C:\Users\USER\我的雲端硬碟\colab_高曼_test\set1\set3_2020-12-31_20.pickle', 'rb') as f:
        tt = pickle.load( f)

    transta = TransTa(typename = 'test')
    transta.input_tt(tt)
    feature = transta.transform(X)
    tail = feature.tail(10000)
    # with open(r'transta.pickle', 'wb') as f:
    #     pickle.dump(transta, f)


    # with open(r'transta.pickle', 'rb') as f:
    #     transta = pickle.load( f)


