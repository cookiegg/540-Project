import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot, to_graph
from keras.regularizers import l2, activity_l2
import copy

def predict_model(model,setpoint,time_steps,utest_start,nntest_start,lstm_length,ykstack):
    for i in range(1,time_steps+1): #Predicting next 100 points with lstm and model
        uhat=model.predict([utest_start, nntest_start], verbose=1)
        ustack=np.append(utest_start,uhat)
        ustack=ustack.reshape(1,len(ustack),1)

        #tacking u[k-3] from ustack
        uk_3=ustack[0,(lstm_length-5),0]
        yk=nntest_start[0,1]

        yk1=0.6*yk+1.2*0.05*uk_3
        ykstack[i-1]=yk1

        utest1=ustack[0,1:lstm_length,0] #updating recent 12 points
        utest_start=utest1.reshape(1,len(utest1),1) #converting to list of list of list

        nntest_start=np.array([1,yk1,setpoint])
        nntest_start=nntest_start.reshape(1,3)

    return ykstack