import pandas as pd
import statsmodels.api as sm
from scipy.stats.distributions import t
from scipy.stats import probplot, jarque_bera
import logging
import numpy as np

def show_moments(time_series):
    try:
        assert type(time_series) == pd.DataFrame, "a DF"
        for column in time_series.columns:
            print(f"Moments for {column}:"
                  f"Mean: {time_series[column].mean()}"
                  f"Variance: {time_series[column].var()}"
                  f"Standard Deviation: {time_series[column].std()}"
                  f"Skew: {time_series[column].skew()}"
                  f"Kurtosis: {time_series[column].kurtosis()}")
    except AssertionError as err:
        logging.exception(err)
        raise err

def distribution(time_series):
    try:
        assert type(time_series) == pd.DataFrame, "a DF"
        for column in time_series.columns:
            print(f"Distribution tests for: {column}")
            print("t-distribution:")
            params = t.fit(time_series[column].dropna())
            print(params)
            print(f"Jarque-Bera:")
            print(jarque_bera(time_series[column]))
    except AssertionError as err:
        logging.exception(err)
        raise err