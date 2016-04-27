from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot, to_graph
import copy

import predict as pdt
import utils as util
import model as mod

# ------------------------------------- Main Loop --------------------------------------------

lstm_length = 13
[X_train, y_train, X_test, y_test, NN_train, NN_test]=util.get_data("./../data/manySinesWithRef.txt", lstm_length, 1, "./../data/testFile.txt")

# Get rid of Y and REF because lstm doesn't want to train on this
X_train = X_train[:, :, 0:1]
X_test = X_test[:, :, 0:1]

model = util.load_model('./../savedModels/ModelG/model', './../savedModels/ModelG/model_weights')

# # retraining
# my_batch_size = 512
# my_epoch = 5
# start_time = time.time()
# model.fit([X_train, NN_train], y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
# print("Training Time : %s seconds --- \n" % (time.time() - start_time))

# test the model
U_hat = model.predict([X_test, NN_test], verbose=1)
U_hat = U_hat.reshape((len(U_hat)))
loss_and_metrics = model.evaluate([X_test, NN_test], y_test[:, 0])
print "test error is: ", loss_and_metrics

# plot the predicted versus the actual U values
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()

#Testing the model with Plant Model
setpoint=1
time_steps=1000 #Poits to predict in the future
ykstack=np.zeros(shape=(time_steps,1))
utest_start=X_test[0,:,:] #taking first 12 points in the test set to start prediction
utest_start= utest_start.reshape((1,lstm_length-1,1))
nntest_start=NN_test[0]
nntest_start=nntest_start.reshape(1,3) #3 denotes 3 dimensions

 #array which has predicted values
ykstack=pdt.predict_model(model,setpoint,time_steps,utest_start,nntest_start,lstm_length,ykstack)
#plotting points predicted with lstm and model
plt.plot(ykstack)
plt.show()