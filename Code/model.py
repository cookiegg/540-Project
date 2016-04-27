from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot, to_graph
from keras.regularizers import l2, activity_l2
import copy

# Function to design and construct the model
# Inputs:
#   1. lstm_data_dim: Dimension of lstm training data, in this case 3 because U, Y, ref
#   2. nn_data_dim: Dimension of NN training data, in this case 3 because Yk, Ref_k, 1
#   3. timesteps: length of LSTM
# Outputs:
#   1. Keras model
def design_model(lstm_data_dim, nn_data_dim, timesteps):
    model_A = Sequential()
    model_B = Sequential()
    model_Combine = Sequential()

    # LSTM Part
    lstm_hidden_size = [20, 100]
    drop_out_rate = [0.5, 0.5]
    model_A.add(LSTM(lstm_hidden_size[0], return_sequences=True, input_shape=(timesteps, lstm_data_dim)))
    model_A.add(Dropout(drop_out_rate[0]))  # return_sequences=True means output cell state C at each LSTM sequence
    model_A.add(LSTM(lstm_hidden_size[1], return_sequences=False))
    model_A.add(Dropout(drop_out_rate[1]))  # return_sequence=False means output only last cell state C in LSTM sequence
    model_A.add(Dense(1, activation='linear'))

    # NN Part
    nn_hidden_size = [20, 20]
    nn_drop_rate = [0.2, 0.2]
    model_B.add(Dense(nn_hidden_size[0], input_dim=nn_data_dim))
    model_B.add(Dropout(nn_drop_rate[0]))
    model_B.add(Dense(nn_hidden_size[1]))
    model_B.add(Dropout(nn_drop_rate[1]))
    model_B.add(Dense(1, activation='linear'))

    # Merge and Final Layer
    model_Combine.add(Merge([model_A, model_B], mode='concat'))
    model_Combine.add(Dense(1, activation='linear'))

    # output the model to a PNG file for visualization
    print "Outputting model graph to model.png"
    graph = to_graph(model_Combine, show_shape=True)
    graph.write_png("model.png")

    return model_Combine

def design_model_A(lstm_data_dim, nn_data_dim, timesteps):
    model_A = Sequential()
    model_B = Sequential()
    model_Combine = Sequential()

    # LSTM Part
    lstm_hidden_size = [40, 100]
    drop_out_rate = [0.6, 0.5]
    reg = [0.01]
    areg = [0.01]
    # unfortunately regularization is not implemented for LSTMs
    model_A.add(LSTM(lstm_hidden_size[0], return_sequences=True, input_shape=(timesteps, lstm_data_dim)))
    model_A.add(Dropout(drop_out_rate[0]))  # return_sequences=True means output cell state C at each LSTM sequence
    model_A.add(LSTM(lstm_hidden_size[1], return_sequences=False))
    model_A.add(Dropout(drop_out_rate[1]))  # return_sequence=False means output only last cell state C in LSTM sequence
    model_A.add(Dense(1, activation='linear', W_regularizer=l2(reg[0]), activity_regularizer=activity_l2(areg[0])))

    # NN Part
    nn_hidden_size = [40, 40]
    nn_drop_rate = [0.5, 0.5]
    nn_reg = [0.01, 0.01, 0.01]
    nn_areg = [0.01, 0.01, 0.01]
    model_B.add(Dense(nn_hidden_size[0], input_dim=nn_data_dim, W_regularizer=l2(nn_reg[0]), activity_regularizer=activity_l2(nn_areg[0])))
    model_B.add(Dropout(nn_drop_rate[0]))
    model_B.add(Dense(nn_hidden_size[1], W_regularizer=l2(nn_reg[1]), activity_regularizer=activity_l2(nn_areg[1])))
    model_B.add(Dropout(nn_drop_rate[1]))
    model_B.add(Dense(1, activation='linear', W_regularizer=l2(nn_reg[2]), activity_regularizer=activity_l2(nn_areg[2])))

    # Merge and Final Layer
    model_Combine.add(Merge([model_A, model_B], mode='concat'))
    model_Combine.add(Dense(1, activation='linear'))

    # output the model to a PNG file for visualization
    print "Outputting model graph to model.png"
    graph = to_graph(model_Combine, show_shape=True)
    graph.write_png("model.png")

    return model_Combine

def design_model_nn(lstm_data_dim, nn_data_dim, timesteps):
    model_B = Sequential()

    # NN Part
    nn_hidden_size = [50, 50]
    nn_drop_rate = [0.4, 0.4]
    nn_reg = [0.01, 0.01, 0.01]
    nn_areg = [0.01, 0.01, 0.01]
    model_B.add(Dense(nn_hidden_size[0], input_dim=nn_data_dim, W_regularizer=l2(nn_reg[0]), activity_regularizer=activity_l2(nn_areg[0])))
    model_B.add(Dropout(nn_drop_rate[0]))
    model_B.add(Dense(nn_hidden_size[1], W_regularizer=l2(nn_reg[1]), activity_regularizer=activity_l2(nn_areg[1])))
    model_B.add(Dropout(nn_drop_rate[1]))
    model_B.add(Dense(1, activation='linear', W_regularizer=l2(nn_reg[2]), activity_regularizer=activity_l2(nn_areg[2])))

    # output the model to a PNG file for visualization
    print "Outputting model graph to model.png"
    graph = to_graph(model_B, show_shape=True)
    graph.write_png("model.png")

    return model_B

def design_model_lstm(lstm_data_dim, nn_data_dim, timesteps):
    model_A = Sequential()

    # LSTM Part
    lstm_hidden_size = [20, 100]
    drop_out_rate = [0.5, 0.5]
    reg = [0.01]
    areg = [0.01]
    # unfortunately regularization is not implemented for LSTMs
    model_A.add(LSTM(lstm_hidden_size[0], return_sequences=True, input_shape=(timesteps, lstm_data_dim)))
    model_A.add(Dropout(drop_out_rate[0]))  # return_sequences=True means output cell state C at each LSTM sequence
    model_A.add(LSTM(lstm_hidden_size[1], return_sequences=False))
    model_A.add(Dropout(drop_out_rate[1]))  # return_sequence=False means output only last cell state C in LSTM sequence
    model_A.add(Dense(1, activation='linear', W_regularizer=l2(reg[0]), activity_regularizer=activity_l2(areg[0])))

    # output the model to a PNG file for visualization
    print "Outputting model graph to model.png"
    graph = to_graph(model_A, show_shape=True)
    graph.write_png("model.png")

    return model_A