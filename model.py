import pandas as pd
import os

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam
from keras.models import load_model

"""
THE CONFIGURATION IN TRAINING THE MODELS IS LISTED HERE
"""
class Model():
    def __init__(self, name, model, n_forecast, n_window, n_feature):
        self.name = name
        self.model = model
        self.n_forecast = n_forecast
        self.n_window = n_window
        self.n_feature = n_feature

    def load_model(self):
        model = load_model(self.model)
        return model
        
def main():
    m = Model("LSTM", "model/LSTM_5days.h5", 30, 45, 4)
    m.load_and_predict_data()

if __name__ == "__main__":
    main()
