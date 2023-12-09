import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    p_value = dftest[1]
    return p_value < 0.05

def kpss_test(timeseries):
    statistic, p_value, n_lags, critical_values = kpss(timeseries, regression='c')
    return p_value > 0.05

def pp_test(timeseries):
    dftest = adfuller(timeseries, autolag=None, maxlag=0)
    p_value = dftest[1]
    return p_value < 0.05

def is_stationary(timeseries):
    return adf_test(timeseries) or kpss_test(timeseries) or pp_test(timeseries)

# # Sample usage with some made-up data
# if __name__ == '__main__':
#     timeseries = pd.Series([1,2,3,4,5,6,7,8,9,10])  # Replace with your time series data
#     print(is_stationary(timeseries))
