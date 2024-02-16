import pandas as pd
import os
import yfinance as yf
import numpy as np
import streamlit as st
import time

"""
IMPORTANT:

DATASET TO TRAIN LSTM_5DAYS AND LSTM_30DAYS MODEL ARE "25-1 Variable Present" IN dataset FOLDER. THE DATA IS IN STANDARDIZED FORMAT
ITS MEAN AND STD IS PROVIDED. 
**USED THEM TO CONVERT BACK WHEN TO ORIGINAL DATA WHEN PREDICTING FUTURE DATA**

"""
DATA_FILE = "dataset/data.csv"
MEAN_FILE = "dataset/mean.csv"
STD_FILE = "dataset/std.csv"

@st.cache_data()
def load_csv():
    # st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
    time.sleep(2)  # This makes the function take 2s to run

    mean_s = pd.read_csv(MEAN_FILE, index_col=0, header=None)
    std_s = pd.read_csv(STD_FILE, index_col=0, header=None)
    data_s = pd.read_csv(DATA_FILE, index_col=0)

    return mean_s, std_s, data_s

@st.cache_data()
def load_data(symbol, period, today):
    time.sleep(2)
    sm = yf.download(symbol, period=period)
    print(f"sm data is: {sm}")
    return sm

# def 


class Database():
    """
    df - dataframe that is fed in the model (in standardized form, (x-mean)/std)
    mean_s - series of means including all column in df
    std_s - series of standard deviation including all column in df
    """
    def __init__(self):
        self.mean_s = None
        self.std_s = None
        self.df = None
        self.mean_s, self.std_s, self.df = load_csv()

    def get_mean_and_std(self, name="Price US Soybean Meal"):
        mean = self.mean_s.loc[name].values[0]
        std = self.std_s.loc[name].values[0]

        # print(f"\nMean of {name}: {mean}\nStd of {name}: {std}\n")  

        return mean, std      

    def print(self):
        print(f"{self.mean_s} \n{self.std_s} \n{self.df}")

    # Standardized scaling
    def standardize_data(self, df, mean, std):
        standardize_data = (df - mean) / std
        return standardize_data

    def standard_undo(self, df, mean, std):
        data_standardization_undo = df * std + mean
        return data_standardization_undo

    def fill_na(self, df):
        df['lag1'] = df['lag1'].bfill()
        df['lag2'] = df['lag2'].bfill()
        df['lag3'] = df['lag3'].bfill()

        df['lag1'] = df['lag1'].ffill()
        df['lag2'] = df['lag2'].ffill()
        df['lag3'] = df['lag3'].ffill()
        return df

    def add_lags(self, prediction_df):
        df = self.df
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y")
        target_map = df['Price US Soybean Meal'].to_dict()
        # print(target_map)
        # print(f"type of df is {prediction_df.index.astype}")
        # print((prediction_df.index - pd.Timedelta('365 days')))
        # print((prediction_df.index - pd.Timedelta('365 days')).map(target_map))

        prediction_df['lag1'] = (prediction_df.index - pd.Timedelta('365 days')).map(target_map)
        prediction_df['lag2'] = (prediction_df.index - pd.Timedelta('730 days')).map(target_map)
        prediction_df['lag3'] = (prediction_df.index - pd.Timedelta('1095 days')).map(target_map)

        prediction_df = self.fill_na(prediction_df)

        return prediction_df

    def add_future_lags(self, n_forecast, prediction_df):
        """
        Add future lags of up to n_forecast day(s)

        Output: extend_df that contains NaN Close value with lags of years
        """
        df = self.df
        df.index = pd.to_datetime(df.index, format="%d/%m/%Y")
        target_map = df['Price US Soybean Meal'].to_dict()

        #  Create timestamp of 10 days
        start_time = prediction_df.index[-1] + pd.Timedelta('1 days')
        end_time = start_time + pd.Timedelta(f'{n_forecast-1} days')
        date_range = pd.date_range(start=start_time, end=end_time)

        extend_df = pd.DataFrame(index=date_range)
        extend_df['Close'] = np.nan
        extend_df['lag1'] = (extend_df.index - pd.Timedelta('365 days')).map(target_map)
        extend_df['lag2'] = (extend_df.index - pd.Timedelta('730 days')).map(target_map)
        extend_df['lag3'] = (extend_df.index - pd.Timedelta('1095 days')).map(target_map)
        extend_df = self.fill_na(extend_df)

        # print(extend_df)
        return extend_df
    
    def extract_data_from_yfinance(self, symbol, period, today):
        sm = load_data(symbol, period, today)
        return sm
    
    def form_prediction_set():
        pass

def main():
    db = Database()
    mean, std = db.get_mean_and_std()
    sm = db.extract_data_from_yfinance("ZMH24.CBT", 'max')
    sm = sm[-45:]
    sm_close = sm['Close']
    sm_close_stz = db.standardize_data(sm_close, mean, std)
    data = pd.DataFrame(sm_close_stz)
    data = db.add_lags(data)
    # print(data)
    extend_data = db.add_future_lags(30, data)

    # print(extend_data.shape)
if __name__ == "__main__":
    main()