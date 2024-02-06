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

import matplotlib.pyplot as plt
import plotly.express as px

from prophet import Prophet

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
    print("Done refreshing")

def predict_LSTM_10days(db):
    """
    Model format: name, model, n_forecast, n_window, n_feature
    """
    st.subheader("Daily Forecasting")
    st.markdown("_Note: Model is update every 1 hour_")
 
    mean, std = db.get_mean_and_std()

    # wtab1, wtab2, wtab3, wtab4, wtab5, wtab6 = st.tabs(["45days", "7days", "1month", "3months", "6months", "1year"])
    # wtab1 = st.tabs(["45days"])

    # with wtab1:
    trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 45, 4)
    model = trained_model.load_model()
    window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')
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
    st.line_chart(combine)  

    st.markdown(f"The result showing a trend of {result}")
    # with wtab2:
    #     trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 7, 4)
    #     model = trained_model.load_model()
    #     window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')

    #     # print(window_normal_data)
    #     reference_start = window_normal_data.index[0]
    #     reference_end = window_normal_data.index[-1]

    #     # Prediction
    #     prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    #     # print(f"Predictions: {prediction}")
    #     prediction = pd.Series(prediction)
    #     actual_prediction = db.standard_undo(prediction, mean, std)
    #     # print(f"Actual Prediction: {actual_prediction}")
    #     actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    #     actual_prediction.index = range(1, len(actual_prediction)+1)

    #     latest_his_date = window_normal_data.index[-1]
    #     pred = actual_prediction.copy()

    #     start = latest_his_date + pd.Timedelta('1 days')
    #     end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    #     pred.index = pd.date_range(start= start, end = end)

    #     combine = pd.concat([window_normal_data, pred], axis=1)
    #     combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]

    #     result = ""
    #     difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    #     if difference > 0:
    #         result = ":green[INCREASING]"
    #     else:
    #         result = ":red[DECREASING]"
    #     # print(actual_prediction)
    #     st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    #     st.line_chart(combine)  

    #     st.markdown(f"The result showing a trend of {result}")
    # with wtab3:
    #     trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 22, 4)
    #     model = trained_model.load_model()
    #     window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')
    #     # print(window_normal_data)
    #     reference_start = window_normal_data.index[0]
    #     reference_end = window_normal_data.index[-1]

    #     # Prediction
    #     prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    #     # print(f"Predictions: {prediction}")
    #     prediction = pd.Series(prediction)
    #     actual_prediction = db.standard_undo(prediction, mean, std)
    #     # print(f"Actual Prediction: {actual_prediction}")
    #     actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    #     actual_prediction.index = range(1, len(actual_prediction)+1)

    #     latest_his_date = window_normal_data.index[-1]
    #     pred = actual_prediction.copy()

    #     start = latest_his_date + pd.Timedelta('1 days')
    #     end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    #     pred.index = pd.date_range(start= start, end = end)

    #     combine = pd.concat([window_normal_data, pred], axis=1)
    #     combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]

    #     result = ""
    #     difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    #     if difference > 0:
    #         result = ":green[INCREASING]"
    #     else:
    #         result = ":red[DECREASING]"
    #     # print(actual_prediction)
    #     st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    #     st.line_chart(combine)  

    #     st.markdown(f"The result showing a trend of {result}")
    # with wtab4:
    #     trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 65, 4)
    #     model = trained_model.load_model()
    #     window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')
    #     # print(window_normal_data)
    #     reference_start = window_normal_data.index[0]
    #     reference_end = window_normal_data.index[-1]

    #     # Prediction
    #     prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    #     # print(f"Predictions: {prediction}")
    #     prediction = pd.Series(prediction)
    #     actual_prediction = db.standard_undo(prediction, mean, std)
    #     # print(f"Actual Prediction: {actual_prediction}")
    #     actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    #     actual_prediction.index = range(1, len(actual_prediction)+1)

    #     latest_his_date = window_normal_data.index[-1]
    #     pred = actual_prediction.copy()

    #     start = latest_his_date + pd.Timedelta('1 days')
    #     end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    #     pred.index = pd.date_range(start= start, end = end)

    #     combine = pd.concat([window_normal_data, pred], axis=1)
    #     combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]

    #     result = ""
    #     difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    #     if difference > 0:
    #         result = ":green[INCREASING]"
    #     else:
    #         result = ":red[DECREASING]"
    #     # print(actual_prediction)
    #     st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    #     st.line_chart(combine)  

    #     st.markdown(f"The result showing a trend of {result}")
    # with wtab5:
    #     trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 130, 4)
    #     model = trained_model.load_model()
    #     window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')
    #     # print(window_normal_data)
    #     reference_start = window_normal_data.index[0]
    #     reference_end = window_normal_data.index[-1]

    #     # Prediction
    #     prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    #     # print(f"Predictions: {prediction}")
    #     prediction = pd.Series(prediction)
    #     actual_prediction = db.standard_undo(prediction, mean, std)
    #     # print(f"Actual Prediction: {actual_prediction}")
    #     actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    #     actual_prediction.index = range(1, len(actual_prediction)+1)

    #     latest_his_date = window_normal_data.index[-1]
    #     pred = actual_prediction.copy()

    #     start = latest_his_date + pd.Timedelta('1 days')
    #     end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    #     pred.index = pd.date_range(start= start, end = end)

    #     combine = pd.concat([window_normal_data, pred], axis=1)
    #     combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]

    #     result = ""
    #     difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    #     if difference > 0:
    #         result = ":green[INCREASING]"
    #     else:
    #         result = ":red[DECREASING]"
    #     # print(actual_prediction)
    #     st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    #     st.line_chart(combine)  

    #     st.markdown(f"The result showing a trend of {result}")
    # with wtab6:
    #     trained_model = Model("LSTM_10days", "model/LSTM_10days.h5", 10, 260, 4)
    #     model = trained_model.load_model()
    #     window_size_data, extend_data, window_normal_data = LSTM_10days.prediction_data_for_LSTM_10days(db, trained_model.n_forecast, trained_model.n_window, "ZMH24.CBT", '1y')
    #     # print(window_normal_data)
    #     reference_start = window_normal_data.index[0]
    #     reference_end = window_normal_data.index[-1]

    #     # Prediction
    #     prediction = LSTM_10days.predict(model, window_size_data, extend_data)
    #     # print(f"Predictions: {prediction}")
    #     prediction = pd.Series(prediction)
    #     actual_prediction = db.standard_undo(prediction, mean, std)
    #     # print(f"Actual Prediction: {actual_prediction}")
    #     actual_prediction = pd.DataFrame(actual_prediction, columns=["Prediction"])
    #     actual_prediction.index = range(1, len(actual_prediction)+1)

    #     latest_his_date = window_normal_data.index[-1]
    #     pred = actual_prediction.copy()

    #     start = latest_his_date + pd.Timedelta('1 days')
    #     end = latest_his_date + pd.Timedelta(f"{len(pred)} days")

    #     pred.index = pd.date_range(start= start, end = end)

    #     combine = pd.concat([window_normal_data, pred], axis=1)
    #     combine.iloc[len(window_normal_data)-1, 1] = combine.iloc[len(window_normal_data)-1, 0]

    #     result = ""
    #     difference = actual_prediction["Prediction"].iloc[-1] - actual_prediction["Prediction"].iloc[0]
    #     if difference > 0:
    #         result = ":green[INCREASING]"
    #     else:
    #         result = ":red[DECREASING]"
    #     # print(actual_prediction)
    #     st.markdown(f"Prediction of next {trained_model.n_forecast} days based on :orange[{reference_start}] to :orange[{reference_end}]")
    #     st.line_chart(combine)  

    #     st.markdown(f"The result showing a trend of {result}")




def get_history_data(db, date=None):
    """
    Return data from historical data to the date specified, if None it will be up to the latest date.
    """
    # print("check")
    # print(db.df)
    history_df = db.df
    his_df = history_df
    history_df.index = pd.to_datetime(history_df.index, format="%d/%m/%Y")
    if db.df["Price US Soybean Meal"].mean() < 10:
        mean, std = db.get_mean_and_std()
        his_df = db.standard_undo(history_df["Price US Soybean Meal"], mean, std)
    else:
        his_df = his_df["Price US Soybean Meal"]
    if date is not None:
        his_df = his_df[his_df.index > date]

    return his_df

def get_history_to_latest_data(db, hist_data):
    """
    combine date from earliest of "hist_data" to latest of "db"
    """
    latest_data = hist_data.index[-1]


    sm = db.extract_data_from_yfinance("ZMH24.CBT", '1y')
    # print(sm)
    # print(latest_data)

    update_data = sm[sm.index > latest_data]
    update_df = update_data['Close']
    # print("this is")
    # print(hist_data)
    # print("and this is")
    # print(update_df)
    combined_series = pd.concat([hist_data, update_df], axis=0)

    combined_df = pd.DataFrame(columns = ["Close Price"])
    combined_df['Close Price'] = combined_series

    # print(combined_df)

    return combined_df

def predict_prophet(historical_to_latest_data):
    st.text("")
    st.subheader("Yearly Forecasting")

    prophet = historical_to_latest_data

    input_data = pd.DataFrame(historical_to_latest_data)
    # print(input_data)
    input_data = input_data.reset_index()
    input_data = input_data.rename(columns={'index': 'Date'})
    # print(input_data)
    print(input_data["Date"].astype)
    input_data = input_data.rename(columns={'Date': 'ds', 'Close Price': 'y'})
    # # Predict wif prophet
    model = Prophet(interval_width=0.95)
    model.fit(input_data)

    future_dates = model.make_future_dataframe(periods=260, freq='B')
    # print(future_dates)

    forecast = model.predict(future_dates)
    # forecast[['ds', 'y', 'yhat_lower', 'yhat_upper']].head()
    fc = forecast[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'yhat']]
    fc.set_index(fc.ds, inplace=True)
    fc = fc.iloc[:, 1:]
    fc = fc.iloc[-260:, :]

    dtab1, dtab2, dtab3 = st.tabs(["Day", "Week", "Month"])

    with dtab1:
        """
        Daily filter
        """
        year = fc.copy()
        print(year)
        st.line_chart(year)
    with dtab2:
        """
        Weekly filter
        """
        week = fc.copy()
        week["Year"] = week.index.year
        week.index = pd.to_datetime(week.index)
        week = week.reset_index()
        print(week["ds"].dt.isocalendar().week.apply(lambda x: "0" + str(x) if len(str(x))==1 else str(x)))
        week["Week"] = week["ds"].dt.isocalendar().week.apply(lambda x: "0" + str(x) if len(str(x))==1 else str(x))

        # print(week)
        fc_week = week.groupby(["Year", "Week"], as_index=False).mean()
        fc_week["Combined Week"] = fc_week["Year"].astype(str) + '-' + fc_week["Week"].astype(str)
        # print(fc_week)
        fc_week = fc_week.iloc[1:, :]
        # print(fc_week)

        fc_week.set_index(["Combined Week"], inplace=True, drop=True)
        fc_week = fc_week.iloc[:, 3:]
        # print(fc_week)
        # fc_week.index = pd.to_datetime(fc_week.index, format='%Y-%W')
        st.line_chart(fc_week, use_container_width=True)
        # print(fc_week)

    with dtab3:
        """
        Monthly filter
        """
        month = fc.copy()
        month["Year"] = month.index.year
        month["Month"] = month.index.month
        fc_month = month.groupby(["Year", "Month"], as_index=False).mean()
        # fc_month = fc_month.set_index(["Year", "Month"])
        fc_month["Combined Date"] = fc_month["Year"].astype(str) + '-' + fc_month["Month"].astype(str)
        fc_month.set_index(["Combined Date"], inplace=True, drop=True)
        fc_month = fc_month.iloc[:, 2:]
        fc_month.index = pd.to_datetime(fc_month.index, format='%Y-%m')
        # print(fc_month)
        st.line_chart(fc_month, use_container_width=True)

    price_max = year["yhat"].max()
    date_max = str(year.index[year["yhat"] == price_max].values[0]).split("T")[0]
    print("thisshi")
    print(str(year.index[year["yhat"] == price_max].values[0]).strip("T"))
    price_min = year["yhat"].min()
    date_min = str(year.index[year["yhat"] == price_min].values[0]).split("T")[0]
    print(f"{price_max}")

    print(f"{date_max}")
    st.markdown(f"Based on the prediction, the highest price predicted are :green[{price_max.round(1)} ({date_max})]; while the lowest price predicted are :red[{price_min.round(1)} ({date_min})]")


def display_yearly(db):
    print("Displaying yearly data")
    st.subheader("Price by Month")

    historical_data = get_history_data(db)
    historical_to_latest_data = get_history_to_latest_data(db, historical_data)

    print(historical_to_latest_data.index.astype)

    years = historical_to_latest_data.index.year
    
    historical_to_latest_data["Yearly"] = years
    historical_to_latest_data["Monthly"] = historical_to_latest_data.index.month

    tab1, tab2 = st.tabs(["Price by Last Day of Month", "Price by its Average of Month"])

    with tab1:
        data_by_end_of_month = (
            historical_to_latest_data.groupby(["Yearly", "Monthly"], as_index=False)["Close Price"].last()
        )

        fig_price_by_month = px.line(
            data_by_end_of_month,
            x="Monthly",
            y="Close Price",
            color="Yearly",
        )

        fig_price_by_month.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )

        data = historical_to_latest_data.groupby(["Yearly", "Monthly"])["Close Price"].last().round(1).unstack()
        data.loc[1995, 1] = data.loc[1994, 12]

        st.dataframe(data)
        st.plotly_chart(fig_price_by_month, theme=None, use_container_width=True)

    with tab2:
        data_by_end_of_month = (
            historical_to_latest_data.groupby(["Yearly", "Monthly"], as_index=False)["Close Price"].mean()
        )

        fig_price_by_month = px.line(
            data_by_end_of_month,
            x="Monthly",
            y="Close Price",
            color="Yearly",
        )

        fig_price_by_month.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )

        data = historical_to_latest_data.groupby(["Yearly", "Monthly"])["Close Price"].mean().round(1).unstack()
        data.loc[1995, 1] = data.loc[1994, 12]

        st.dataframe(data)
        st.plotly_chart(fig_price_by_month, theme=None, use_container_width=True)

def main():
    db = Database()
    # model = Model()

    historical_data = get_history_data(db, "01/01/2015")
    all_historical_data = get_history_data(db)
    historical_to_latest_data = get_history_to_latest_data(db, all_historical_data)
    historical_to_latest_data_2015 = get_history_to_latest_data(db, historical_data)

    latest_date = gather_basic_information(db)  
    st.title("Time Series Analysis")
    st.subheader("This website forecasts time series data using machine learning approach: LSTM and Prophet.")
    st.markdown(f"_The model is updated up to {latest_date}_") # see *

    st.markdown(f"Historical Data of Soybean Meal since 2015")

    # Display historical data
    st.line_chart(historical_to_latest_data_2015, y="Close Price")

    # Diplay data by year
    display_yearly(db)

    # Daily forecasting 
    predict_LSTM_10days(db)

    # Yearly forecasting
    predict_prophet(historical_to_latest_data)

    exit_app = st.sidebar.button("Shut Down")
    refresh_app = st.sidebar.button("Refresh")

    if exit_app:
        quit()

    if refresh_app:
        refresh()


if __name__ == "__main__":
    main()