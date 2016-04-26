from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout,Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import copy




data_dim = 1
timesteps = 13

# expected input data shape: (batch_size, timesteps, data_dim)
model_A = Sequential()
model_B = Sequential()
model_Combine = Sequential()


model_A.add(LSTM(10, input_shape=(timesteps, data_dim)))
model_A.add(Dense(1, activation='linear'))
model_B.add(Dense(10, input_dim=12))
model_B.add(Dense(1, activation='linear'))

model_Combine.add(Merge([model_A, model_B], mode='concat'))


model_Combine.add(Dense(1, activation='linear'))

model_Combine.compile(loss="mse", optimizer="rmsprop")


# output the model
from keras.utils.visualize_util import plot, to_graph
graph = to_graph(model_Combine, show_shape=True)
graph.write_png("rnnandnn.png")
