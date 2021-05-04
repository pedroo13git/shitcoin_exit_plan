import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging


def make_returns(time_series):
    try:
        assert type(time_series) == pd.DataFrame, "a dataframe, A DATAFRAME!"
        data = np.log(time_series) - np.log(time_series.shift(1))
        return data
    except AssertionError as err:
        logging.exception(err)
        raise err


def get_ticker(ticker_list, start="2000-01-01", end=None, log_returns=False):
    try:
        assert type(ticker_list) == list, "list means [ticker1, ticker2,...]"
        # assert type(start) == str or None, "like this 'yyyy-mm-dd'"
        # assert type(end) == str or None, "like this 'yyyy-mm-dd'"
        assert type(log_returns) == bool, "True or False, don't even know how you got here"
        lista_data = []
        for i in ticker_list:
            tickerData = yf.Ticker(i)
            lista_data.append(tickerData.history(period='1d', start=start, end=end))

        data_price = pd.DataFrame(columns=ticker_list).fillna(method='ffill').fillna(method='bfill')
        for i, val in enumerate(lista_data):
            data_price[data_price.columns[i]] = val['Close']

        if log_returns == True:
            data = np.log(data_price) - np.log(data_price.shift(1))
            return data
        else:
            return data_price
    except AssertionError as err:
        logging.exception(err)
        raise err


def scaler_timeseries(time_series):
    try:
        assert type(time_series) == pd.DataFrame, "a pandas dataframe that is time series in columns, pls"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(X=time_series.to_numpy().reshape(-1, 1))
        data_scaled_indexed = pd.Series(scaled_data.reshape(1, -1)[0], index=time_series.index)
        return data_scaled_indexed
    except AssertionError as err:
        logging.exception(err)
        raise err


def historical_mean_and_cov(time_series):
    try:
        assert type(time_series) == pd.DataFrame, "a pandas dataframe is required"
        mean = time_series.mean()
        covariance = time_series.cov()
        return mean, covariance
    except AssertionError as err:
        logging.exception(err)
        raise err
