# Series data
tune_series = ["open", "high", "low", "close", "volume"]

# testing
test = ['???']

test2 = ['XXXX']

# Parameters to tune
tune_params = [
    "acceleration",
    "accelerationlong",
    "accelerationshort",
    "atr_length",
    "atr_period",
    "average_lenght",
    "average_length",
    "bb_length",
    "channel_lenght",
    "channel_length",
    "chikou_period",
    "d",
    "ddof",
    "ema_fast",
    "ema_slow",
    "er",
    "fast",
    "fast_period",
    "fastd_period",
    "fastk_period",
    "fastperiod",
    "high_length",
    "jaw",
    "k",
    "k_period",
    "kc_length",
    "kijun",
    "kijun_period",
    "length",
    "lensig",
    "lips",
    "long",
    "long_period",
    "lookback",
    "low_length",
    "lower_length",
    "lower_period",
    "ma_length",
    "mamode",
    "max_lookback",
    "maxperiod",
    "medium",
    "min_lookback",
    "minperiod",
    "mom_length",
    "mom_smooth",
    "p",
    "period",
    "period_fast",
    "period_slow",
    "q",
    "r1",
    "r2",
    "r3",
    "r4",
    "roc1",
    "roc2",
    "roc3",
    "roc4",
    "rsi_length",
    "rsi_period",
    "run_length",
    "rvi_length",
    "senkou",
    "senkou_period",
    "short",
    "short_period",
    "signal",
    "signalperiod",
    "slow",
    "slow_period",
    "slowd_period",
    "slowk_period",
    "slowperiod",
    "sma1",
    "sma2",
    "sma3",
    "sma4",
    "smooth",
    "smooth_k",
    "stoch_period",
    "swma_length",
    "tclength",
    "teeth",
    "tenkan",
    "tenkan_period",
    "timeperiod",
    "timeperiod1",
    "timeperiod2",
    "timeperiod3",
    "upper_length",
    "upper_period",
    "width",
    "wma_period",
]

talib_indicators = [
    "tta.BBANDS",
    "tta.DEMA",
    "tta.EMA",
    "tta.HT_TRENDLINE",
    "tta.KAMA",
    "tta.MA",
    "tta.MIDPOINT",
    "tta.MIDPRICE",
    "tta.SAR",
    "tta.SAREXT",
    "tta.SMA",
    "tta.T3",
    "tta.TEMA",
    "tta.TRIMA",
    "tta.WMA",
    "tta.ADX",
    "tta.ADXR",
    "tta.APO",
    "tta.AROON",
    "tta.AROONOSC",
    "tta.BOP",
    "tta.CCI",
    "tta.CMO",
    "tta.DX",
    "tta.MACD",
    "tta.MACDEXT",
    "tta.MACDFIX",
    "tta.MFI",
    "tta.MINUS_DI",
    "tta.MINUS_DM",
    "tta.MOM",
    "tta.PLUS_DI",
    "tta.PLUS_DM",
    "tta.PPO",
    "tta.ROC",
    "tta.ROCP",
    "tta.ROCR",
    "tta.ROCR100",
    "tta.RSI",
    "tta.STOCH",
    "tta.STOCHF",
    "tta.STOCHRSI",
    "tta.TRIX",
    "tta.ULTOSC",
    "tta.WILLR",
    "tta.AD",
    "tta.ADOSC",
    "tta.OBV",
    "tta.HT_DCPERIOD",
    "tta.HT_DCPHASE",
    "tta.HT_PHASOR",
    "tta.HT_SINE",
    "tta.HT_TRENDMODE",
    "tta.AVGPRICE",
    "tta.MEDPRICE",
    "tta.TYPPRICE",
    "tta.WCLPRICE",
    "tta.ATR",
    "tta.NATR",
    "tta.TRANGE",
    "tta.CDL2CROWS",
    "tta.CDL3BLACKCROWS",
    "tta.CDL3INSIDE",
    "tta.CDL3LINESTRIKE",
    "tta.CDL3OUTSIDE",
    "tta.CDL3STARSINSOUTH",
    "tta.CDL3WHITESOLDIERS",
    "tta.CDLABANDONEDBABY",
    "tta.CDLADVANCEBLOCK",
    "tta.CDLBELTHOLD",
    "tta.CDLBREAKAWAY",
    "tta.CDLCLOSINGMARUBOZU",
    "tta.CDLCONCEALBABYSWALL",
    "tta.CDLCOUNTERATTACK",
    "tta.CDLDARKCLOUDCOVER",
    "tta.CDLDOJI",
    "tta.CDLDOJISTAR",
    "tta.CDLDRAGONFLYDOJI",
    "tta.CDLENGULFING",
    "tta.CDLEVENINGDOJISTAR",
    "tta.CDLEVENINGSTAR",
    "tta.CDLGAPSIDESIDEWHITE",
    "tta.CDLGRAVESTONEDOJI",
    "tta.CDLHAMMER",
    "tta.CDLHANGINGMAN",
    "tta.CDLHARAMI",
    "tta.CDLHARAMICROSS",
    "tta.CDLHIGHWAVE",
    "tta.CDLHIKKAKE",
    "tta.CDLHIKKAKEMOD",
    "tta.CDLHOMINGPIGEON",
    "tta.CDLIDENTICAL3CROWS",
    "tta.CDLINNECK",
    "tta.CDLINVERTEDHAMMER",
    "tta.CDLKICKING",
    "tta.CDLKICKINGBYLENGTH",
    "tta.CDLLADDERBOTTOM",
    "tta.CDLLONGLEGGEDDOJI",
    "tta.CDLLONGLINE",
    "tta.CDLMARUBOZU",
    "tta.CDLMATCHINGLOW",
    "tta.CDLMATHOLD",
    "tta.CDLMORNINGDOJISTAR",
    "tta.CDLMORNINGSTAR",
    "tta.CDLONNECK",
    "tta.CDLPIERCING",
    "tta.CDLRICKSHAWMAN",
    "tta.CDLRISEFALL3METHODS",
    "tta.CDLSEPARATINGLINES",
    "tta.CDLSHOOTINGSTAR",
    "tta.CDLSHORTLINE",
    "tta.CDLSPINNINGTOP",
    "tta.CDLSTALLEDPATTERN",
    "tta.CDLSTICKSANDWICH",
    "tta.CDLTAKURI",
    "tta.CDLTASUKIGAP",
    "tta.CDLTHRUSTING",
    "tta.CDLTRISTAR",
    "tta.CDLUNIQUE3RIVER",
    "tta.CDLUPSIDEGAP2CROWS",
    "tta.CDLXSIDEGAP3METHODS",
    "tta.LINEARREG",
    "tta.LINEARREG_ANGLE",
    "tta.LINEARREG_INTERCEPT",
    "tta.LINEARREG_SLOPE",
    "tta.STDDEV",
    "tta.TSF",
    "tta.VAR",
]

pandas_ta_indicators = [
    "pta.ao",
    "pta.apo",
    "pta.bias",
    "pta.bop",
    "pta.brar",
    "pta.cci",
    "pta.cfo",
    "pta.cg",
    "pta.cmo",
    "pta.coppock",
    "pta.cti",
    "pta.dm",
    "pta.er",
    "pta.eri",
    "pta.fisher",
    "pta.inertia",
    "pta.kdj",
    "pta.kst",
    "pta.macd",
    "pta.mom",
    "pta.pgo",
    "pta.ppo",
    "pta.psl",
    "pta.qqe",
    "pta.roc",
    "pta.rsi",
    "pta.rsx",
    "pta.rvgi",
    "pta.stc",
    "pta.slope",
    "pta.smi",
    "pta.squeeze",
    "pta.squeeze_pro",
    "pta.stoch",
    # "pta.stochf", # remove when pandas-ta development is merged
    "pta.stochrsi",
    # "pta.td_seq", # because of the following error: The value nan is not acceptable.
    "pta.trix",
    "pta.tsi",
    "pta.uo",
    "pta.willr",
    # "pta.alligator", # remove when pandas-ta development is merged
    "pta.alma",
    "pta.dema",
    "pta.ema",
    "pta.fwma",
    "pta.hilo",
    "pta.hl2",
    "pta.hlc3",
    # "pta.hma", # does not accept low ranges
    "pta.hwma",
    # "pta.ichimoku", all is na
    "pta.jma",
    "pta.kama",
    "pta.linreg",
    # "pta.mama", # remove when pandas-ta development is merged
    "pta.mcgd",
    "pta.midpoint",
    "pta.midprice",
    "pta.ohlc4",
    "pta.pwma",
    "pta.rma",
    "pta.sinwma",
    "pta.sma",
    # "pta.smma", # remove when pandas-ta development is merged
    "pta.ssf",
    # "pta.ssf3", # remove when pandas-ta development is merged
    "pta.supertrend",
    "pta.swma",
    "pta.t3",
    "pta.tema",
    "pta.trima",
    "pta.vidya",
    "pta.wcp",
    "pta.wma",
    "pta.zlma",
    "pta.entropy",
    "pta.kurtosis",
    "pta.mad",
    "pta.median",
    "pta.quantile",
    "pta.skew",
    "pta.stdev",
    # "pta.tos_stdevall", # remove when pandas-ta development is merged
    "pta.variance",
    "pta.zscore",
    "pta.adx",
    "pta.amat",
    "pta.aroon",
    "pta.chop",
    "pta.cksp",
    "pta.decay",
    "pta.decreasing",
    "pta.dpo",
    "pta.increasing",
    # "pta.psar",  # The value nan is not acceptable.
    "pta.qstick",
    # "pta.trendflex", # remove when pandas-ta development is merged
    "pta.ttm_trend",
    "pta.vhf",
    "pta.vortex",
    "pta.aberration",
    "pta.accbands",
    "pta.atr",
    # "pta.atrts", # remove when pandas-ta development is merged
    "pta.bbands",
    "pta.donchian",
    # "pta.hwc",  # The value nan is not acceptable.
    "pta.kc",
    "pta.massi",
    "pta.natr",
    "pta.pdist",
    "pta.rvi",
    "pta.thermo",
    "pta.true_range",
    "pta.ui",
    "pta.ad",
    "pta.adosc",
    "pta.aobv",
    "pta.cmf",
    "pta.efi",
    "pta.eom",
    "pta.kvo",
    "pta.mfi",
    "pta.nvi",
    "pta.obv",
    "pta.pvi",
    "pta.pvo",
    "pta.pvol",
    # "pta.pvr", # needs **kwargs added to the function in pandas-ta
    "pta.pvt",
    "pta.vp",
    "pta.vwap",
    # "pta.vwma", # remove when pandas-ta development is merged
    # "pta.wb_tsv", # remove when pandas-ta development is merged
]

pandas_ta_mamodes = {
    "dema": 0,
    "ema": 1,
    "fwma": 2,
    # "hma": 3,  # Issue with low range https://github.com/jmrichardson/tuneta/pull/24
    "linreg": 4,
    "midpoint": 5,
    "pwma": 6,
    "rma": 7,
    "sinwma": 8,
    "sma": 9,
    "swma": 10,
    "t3": 11,
    "tema": 12,
    "trima": 13,
    "vidya": 14,
    "wma": 15,
    "zlma": 16,
}

finta_indicatrs = [
    "fta.SMA",
    "fta.SMM",
    "fta.SSMA",
    "fta.EMA",
    "fta.DEMA",
    "fta.TEMA",
    "fta.TRIMA",
    "fta.TRIX",
    "fta.VAMA",
    "fta.ER",
    "fta.KAMA",
    "fta.ZLEMA",
    "fta.WMA",
    "fta.HMA",
    "fta.EVWMA",
    "fta.VWAP",
    "fta.SMMA",
    # 'fta.FRAMA',  # Requires even parameters (not implemented)
    "fta.MACD",
    "fta.PPO",
    "fta.VW_MACD",
    "fta.EV_MACD",
    "fta.MOM",
    "fta.ROC",
    "fta.RSI",
    "fta.IFT_RSI",
    "fta.TR",
    "fta.ATR",
    "fta.SAR",
    "fta.BBANDS",
    "fta.BBWIDTH",
    "fta.MOBO",
    "fta.PERCENT_B",
    "fta.KC",
    "fta.DO",
    "fta.DMI",
    "fta.ADX",
    "fta.PIVOT",
    "fta.PIVOT_FIB",
    "fta.STOCH",
    "fta.STOCHD",
    "fta.STOCHRSI",
    "fta.WILLIAMS",
    "fta.UO",
    "fta.AO",
    "fta.MI",
    "fta.VORTEX",
    "fta.KST",
    "fta.TSI",
    "fta.TP",
    "fta.ADL",
    "fta.CHAIKIN",
    "fta.MFI",
    "fta.OBV",
    "fta.WOBV",
    "fta.VZO",
    "fta.PZO",
    "fta.EFI",
    "fta.CFI",
    "fta.EBBP",
    "fta.EMV",
    "fta.CCI",
    "fta.COPP",
    "fta.BASP",
    "fta.BASPN",
    "fta.CMO",
    "fta.CHANDELIER",
    "fta.QSTICK",
    # 'fta.TMF',  # Not implemented
    "fta.WTO",
    "fta.FISH",
    "fta.ICHIMOKU",
    "fta.APZ",
    "fta.SQZMI",
    "fta.VPT",
    "fta.FVE",
    "fta.VFI",
    "fta.MSD",
    "fta.STC",
    # 'fta.WAVEPM'  # No attribute error
]
