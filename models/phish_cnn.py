import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import statistics as st


def load_data():
    x = ['hello', 'world', 'the', 'music', 'where', 'dog', 'the', 'time']
    classes = pd.Series(x)

phish_model = keras.Sequential()

load_data()