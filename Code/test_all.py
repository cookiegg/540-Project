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

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#Testing the model with Plant Model  with LSTM and NN

time_steps=1000 #Poits to predict in the future
ykstack=np.zeros(shape=(time_steps,1))
utest_start=X_test[0,:,:] #taking first 12 points in the test set to start prediction
utest_start= utest_start.reshape((1,lstm_length-1,1))
nntest_start=NN_test[0]
nntest_start=nntest_start.reshape(1,3) #3 denotes 3 dimensions

#array which has predicted values
#specify setpoint in predict function
#predict_model for constant set point
#predict_model2 for varying setpoint
ykstack=pdt.predict_model(model,y_test,time_steps,utest_start,nntest_start,lstm_length,ykstack)
#plotting points predicted with lstm and model
plt.plot(ykstack)
plt.show()

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#Testing the model with Plant Model with only LSTM
 #Poits to predict in the future


#varying setpoint (comment this section if you want to use constant setpoint)
yreftest=X_test[:,:,2]
yreftest_comparison=y_test[:,2]
#varying setpoint

#constant setpoint:
setpoint=5
yreftestlen=len(yreftest)
yreftest=setpoint*np.ones(shape=(yreftestlen,4))
yreftest_comparison=setpoint*np.ones(shape=(yreftestlen,1))
#constant setpoint



time_steps=yreftestlen-lstm_length
ykstack=np.zeros(shape=(time_steps,1))


ykstack=pdt.predict_model_lstm(model,X_test,lstm_length,time_steps,ykstack,yreftest,yreftest_comparison)
plt.plot(ykstack)
plt.plot(yreftest_comparison)
plt.show()

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#Testing the model with Plant Model with only NN


# Get rid of the Y's for targets, keep only the U's
u_test = y_test[:, 0]
u_test=u_test.reshape(len(u_test),1)
utest_start=u_test[0:lstm_length-1]



#varying setpoint
#yreftest=NN_test[lstm_length-1:-1,2]
#yreftest_comparison=yreftest.reshape(len(yreftest),1)
#varying setpoint

#constant setpoint:
setpoint=7
yreftestlen=len(NN_test[lstm_length-1:-1,2])
yreftest=setpoint*np.ones(shape=(yreftestlen,1))
yreftest_comparison=yreftest.reshape(yreftestlen,1)
#constant setpoint

time_step=yreftestlen
ykstack=np.zeros(shape=(time_steps-1,1))



NNtest_start=NN_test[lstm_length-2,:]
NNtest_start=NNtest_start.reshape(len(NNtest_start),1)
ytest_start=NNtest_start[1]
yreftest_start=yreftest[0]

bias=1
NNtest_start=np.column_stack((bias,ytest_start))

NNtest_start=np.column_stack((NNtest_start,yreftest_start))

ykstack=pdt.predict_model_nn(model,lstm_length,time_steps,NNtest_start,utest_start,ytest_start,yreftest_start,yreftest,bias,ykstack)


plt.plot(ykstack)
plt.plot(yreftest_comparison)
plt.show()
#------------------------------------------------------------------------------------------






