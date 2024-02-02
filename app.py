import streamlit as st
import time
import keyboard
import psutil
import os
import pandas as pd
import numpy as np

from model import Model
from database import Database
from model_scripts import LSTM_10days as LSTM_10days

# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
# import plotly.offline as py

"""
"""
def gather_basic_information(database, model=1):
    data_file = database.df
    last_date = data_file.index[-1]
    # print(last_date)
    return last_date

def quit():
    """
    button to stop terminate the process.
    """
    # Give a bit of delay for user experience
    time.sleep(1)
    # Close streamlit browser tab
    keyboard.press_and_release('ctrl+w')
    # Terminate streamlit python process
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()

def execute():
    """
    code to execute the prediction.
    1. Get models
    2. Get data for prediction
    3. Get predicted data
    4. Visualize the data
    """
    pass

def refresh():
    """
    refresh the page
    """
    print("IMHERE")

def predict_LSTM_10days(db):
    """
    Model format: name, model, n_forecast, n_window, n_feature
    """
    st.subheader("Daily Forecasting")
    trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 45, 4)
    # print(trained_model.name)
    model = trained_model.load_model()
    # print(trained_model)

    mean, std = db.get_mean_and_std()

    window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '90d')

    # print(window_normal_data)
    reference_start = window_normal_data.index[0]
    reference_end = window_normal_data.index[-1]

    # Prediction
    prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    # print(f"Predictions: {prediction}")
    prediction = pd.Series(prediction)
    actual_prediction = db.standard_undo(prediction, mean, std)
    # print(f"Actual Prediction: {actual_prediction}")
    actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    actual_prediction.index = range(1, len(actual_prediction)+1)

    # print(actual_prediction)

    # Combine data

    latest_his_date = window_normal_data.index[-1]
    pred = actual_prediction.copy()

    start = latest_his_date + pd.Timedelta('1 days')
    end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    pred.index = pd.date_range(start= start, end = end)

    combine = pd.concat([window_normal_data, pred], axis=1)
    combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]


    result = ""
    difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    if difference > 0:
        result = ":green[INCREASING]"
    else:
        result = ":red[DECREASING]"
    # print(actual_prediction)
        
    st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    left_column, right_column = st.columns(2)
    left_column.line_chart(actual_prediction, y=["Prediction"], color="#4169E1")
    right_column.line_chart(combine)  

    # print(prediction.index[0])
    # st.line_chart(prediction, y=["Prediction"])

    st.markdown(f"The result showing a trend of {result}")
    return actual_prediction, result, combine

def get_history_data(db, date=None):
    history_df = db.df
    history_df.index = pd.to_datetime(history_df.index, format="%d/%m/%Y")
    mean, std = db.get_mean_and_std()
    his_df = db.standard_undo(history_df["Price US Soybean Meal"], mean, std)
    if date is not None:
        his_df = his_df[his_df.index > date]
    # return
    # print(his_df)

    return his_df

def get_history_to_latest_data(db, hist_data):
    latest_data = hist_data.index[-1]

    sm = db.extract_data_from_yfinance("ZMH24.CBT", '300d')
    # print(sm)
    # print(latest_data)

    update_data = sm[sm.index > latest_data]
    update_df = update_data['Close']
    # print(hist_data)
    # print(update_df)
    combined_series = pd.concat([hist_data, update_df], axis=0)

    combined_df = pd.DataFrame(columns = ["Close Price"])
    combined_df['Close Price'] = combined_series

    # print(combined_df)

    return combined_df

def predict_prophet(db):
    prophet = pd.read_csv("Prophet.csv", index_col=1)
    prophet = prophet.iloc[:, 1:]
    
    st.text("")
    st.subheader("Yearly Forecasting")
    st.line_chart(prophet)  
    # prophet = prophet.set_index(prophet.Date)
    # print(prophet)

    # historical_data = get_history_data(db)
    # print(historical_data)
    # input_data = pd.DataFrame(historical_data)
    # input_data = input_data.reset_index()
    # input_data = input_data.rename(columns={'index': 'Date'})
    # print(input_data)
    # print(input_data["Date"].astype)

    # # Predict wif prophet
    # model = Prophet(interval_width=0.95)
    # model.fit(input_data)

    # future_dates = model.make_future_dataframe(period=260, freq='B')
    # print(future_dates)

    # forecast = model.predict(future_dates)
    # # forecast[['Date', 'Close', 'yhat_lower', 'yhat_upper']].head()
    # print(forecast)

def main():
    db = Database()
    # model = Model()
    historical_data = get_history_data(db, "01/01/2015")
    historical_to_latest_data = get_history_to_latest_data(db, historical_data)

    latest_date = gather_basic_information(db)  
    st.title("Time Series Analysis")
    st.subheader("This website forecasts time series data using machine learning approach: LSTM and Prophet.")
    st.markdown(f"_The model is updated up to {latest_date}_") # see *

    st.markdown(f"Historical Data since 2015")
    st.line_chart(historical_to_latest_data, y="Close Price")

    predict_LSTM_10days(db)
    predict_prophet(db)
    # print(f"this is {historical_data.iloc[len(historical_data)-1]}")

    # combine["combine"] = combine['Price US Soybean Meal'].fillna(combine['Prediction']).where(~combine['Price US Soybean Meal'].isna(), combine['Prediction'])
    # print(combine.tail(20))

    # st.line_chart(combine)  

    exit_app = st.sidebar.button("Shut Down")
    refresh_app = st.sidebar.button("Refresh")

    if exit_app:
        quit()

    if refresh_app:
        refresh()


if __name__ == "__main__":
    main()