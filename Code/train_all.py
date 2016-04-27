from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot, to_graph
from keras.regularizers import l2, activity_l2
import copy

import utils as util
import model as mod

# ------------------------------------- Main Loop --------------------------------------------
# Get the data
lstm_length = 5;
[X_train, y_train, X_test, y_test, NN_train, NN_test]=util.get_data("./../data/sineAndJump.txt", lstm_length, 1, "./../data/testFile.txt")


# training parameters
my_batch_size = 512
my_epoch = 1
start_time = time.time()

#for only lstm----------------
#model.fit(X_train, y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
#print("Training Time : %s seconds --- \n" % (time.time() - start_time))
#for only lstm----------------

#x=1 lstm
#x=2 NN
#x=3 
x=1


if x == 1:
    print 'Training LSTM'
    # define the input sizes for the LSTM
    lstm_data_dim = X_train.shape[2]
    nn_data_dim = NN_train.shape[1]
    timesteps = lstm_length
    #construct and compile the model
    model = mod.design_model_lstm(lstm_data_dim, nn_data_dim, timesteps)
    start_time = time.time()
    print "Compiling Model ..."
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compile Time : %s seconds --- \n" % (time.time() - start_time))
    
    #learn parameters
    model.fit(X_train, y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
    print("Training Time : %s seconds --- \n" % (time.time() - start_time))
    # test the model
    U_hat = model.predict(X_test, verbose=1)
    U_hat = U_hat.reshape((len(U_hat)))
    loss_and_metrics = model.evaluate(X_test, y_test[:, 0])
    print "test error is: ", loss_and_metrics
elif x==2:
    print 'Training NN'
    # Get rid of Y and REF because lstm doesn't want to train on this
    
    # define the input sizes for the LSTM
    lstm_data_dim = X_train.shape[2]
    nn_data_dim = NN_train.shape[1]
    timesteps = lstm_length

    # construct and compile the model
    model = mod.design_model_nn(lstm_data_dim, nn_data_dim, timesteps)
    start_time = time.time()
    print "Compiling Model ..."
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compile Time : %s seconds --- \n" % (time.time() - start_time))

   
    
    model.fit(NN_train, y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
    print("Training Time : %s seconds --- \n" % (time.time() - start_time))
    # test the model
    U_hat = model.predict(NN_test, verbose=1)
    U_hat = U_hat.reshape((len(U_hat)))
    loss_and_metrics = model.evaluate(NN_test, y_test[:, 0])
    print "test error is: ", loss_and_metrics
    
elif x==3:
    print 'Training LSTM and NN' 
    # Get rid of Y and REF because lstm doesn't want to train on this
    X_train = X_train[:, :, 0:1]
    X_test = X_test[:, :, 0:1]
    
    # define the input sizes for the LSTM
    lstm_data_dim = X_train.shape[2]
    nn_data_dim = NN_train.shape[1]
    timesteps = lstm_length

    # construct and compile the model
    model = mod.design_model(lstm_data_dim, nn_data_dim, timesteps)
    start_time = time.time()
    print "Compiling Model ..."
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compile Time : %s seconds --- \n" % (time.time() - start_time))

   
    
    model.fit([X_train, NN_train], y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
    print("Training Time : %s seconds --- \n" % (time.time() - start_time))
    # test the model
    U_hat = model.predict([X_test, NN_test], verbose=1)
    U_hat = U_hat.reshape((len(U_hat)))
    loss_and_metrics = model.evaluate([X_test, NN_test], y_test[:, 0])
    print "test error is: ", loss_and_metrics
    
    
# plot the predicted versus the actual U values
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()