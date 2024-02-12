import pandas as pd
import numpy as np

def prediction_data_for_LSTM_10days(db, n_forecast, n_window, symbol, time_step, today):
    # Get the mean and std of close price if not specify
    mean, std = db.get_mean_and_std()

    sm = db.extract_data_from_yfinance(symbol, time_step, today)
    print(f"data from yfinance: {sm}")
    sm = sm[-n_window:]
    sm_close = sm['Close']
    # This standardize_data function takes Series, not DataFrame
    sm_close_stz = db.standardize_data(sm_close, mean, std)

    data = pd.DataFrame(sm_close_stz)
    data = db.add_lags(data)
    extend_data = db.add_future_lags(n_forecast, data)

    print(extend_data)

    return data, extend_data, sm_close

def predict(model, window_df, future_df):
    n_forecast = future_df.shape[0]
    # print(window_df.shape)
    # print(future_df.shape)

    test_predictions = []
    first_eval_batch = window_df.to_numpy()
    # print(first_eval_batch)
    # print(window_df.shape[0])
    # print(window_df.shape[1])
    current_batch = first_eval_batch.reshape((1, window_df.shape[0], window_df.shape[1]))
    # print(current_batch)
    for i in range(n_forecast):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred[0])
        # print(current_pred)
        # print(future_df)
        future_lags = future_df.iloc[i, -3:]
        new_batch = np.append(current_pred, future_lags.to_numpy())
        # print(new_batch)
        current_batch = np.append(current_batch[:, 1:, :], [[new_batch]], axis=1)

    return test_predictions