import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot, to_graph
from keras.regularizers import l2, activity_l2
import copy

#lstm + nn for constant setpoint
def predict_model(model,y_test,time_steps,utest_start,nntest_start,lstm_length,ykstack):
    setpoint=5
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

#lstm + nn for varying setpoint    
def predict_model2(model,y_test,time_steps,utest_start,nntest_start,lstm_length,ykstack):
    setpoint= y_test[0:(len(y_test)-1),2]
    
    
    for i in range(1,time_steps+1): #Predicting next 100 points with lstm and model
        uhat=model.predict([utest_start, nntest_start], verbose=1)
        ustack=np.append(utest_start,uhat)
        ustack=ustack.reshape(1,len(ustack),1)

        #tacking u[k-3] from ustack
        uk_3=ustack[0,(lstm_length-5),0]
        yk=nntest_start[0,1]

        yk1=0.6*yk+0.8*0.05*uk_3
        ykstack[i-1]=yk1

        utest1=ustack[0,1:lstm_length,0] #updating recent 12 points
        utest_start=utest1.reshape(1,len(utest1),1) #converting to list of list of list

        nntest_start=np.array([1,yk1,setpoint[i-1]])
        nntest_start=nntest_start.reshape(1,3)

    return ykstack    

#prediction for only lstm
def predict_model_lstm(model,X_test,lstm_length,time_steps,ykstack,yreftest,yreftest_comparison):
    utest_start=X_test[0:1,:,0]
    utest_start=utest_start.reshape((lstm_length-1),(len(utest_start)))
    ytest_start=X_test[0:1,:,1]
    ytest_start=ytest_start.reshape((lstm_length-1),(len(ytest_start)))
    yreftest_start=yreftest[0,:]
    #yreftest_start=yreftest_start.reshape((lstm_length-1),(len(yreftest_start)))
    temp1=np.column_stack((utest_start,ytest_start))
    temp1=np.column_stack((temp1,yreftest_start))
    xtest_start=temp1.reshape(1,lstm_length-1,3)
    #xtest_start=X_test[1:2,:,:]


    ykstack=np.zeros(shape=(time_steps,1))

    for i in range(1,time_steps+1):
        print i
    
        uhat = model.predict(xtest_start, verbose=1)
        ustack=np.append(utest_start,uhat)
        ustack=ustack.reshape(1,len(ustack),1)
        
        uk_3=ustack[0,(lstm_length-5),0]
        yk=ytest_start[lstm_length-2,0]
        
        yk1=0.6*yk+0.05*uk_3
        ykstack[i-1]=yk1
        ystack=np.append(ytest_start,yk1)
        ystack=ystack.reshape(1,len(ystack),1)
        
        utest_start=ustack[0,1:lstm_length,0]
        #utest_start=utest_start.reshape(1,len(utest_start),1)
        ytest_start=ystack[0,1:lstm_length,0]
        ytest_start=ytest_start.reshape((lstm_length-1),(1))
        #ytest_start=ytest_start.reshape(1,len(ytest_start),1)
        temp=np.column_stack((utest_start,ytest_start))
        
    
    
        #xtest_start=np.column_stack((temp,yreftest))
        #xtest_start=xtest_start.reshape(1,(lstm_length-1),3)
        yreftest_start=yreftest[i,:]
        
        xtest_start=np.column_stack((temp,yreftest_start))
        xtest_start=xtest_start.reshape(1,lstm_length-1,3)
    return ykstack        


def predict_model_nn(model,lstm_length,time_steps,NNtest_start,utest_start,ytest_start,yreftest_start,yreftest,bias,ykstack):
    for i in range(1,time_steps):
        print i
    
    
        uhat = model.predict(NNtest_start, verbose=1)
        ustack=np.append(utest_start,uhat)
        ustack=ustack.reshape(1,len(ustack),1)
        uk_3=ustack[0,(lstm_length-5),0]
        yk=ytest_start;
        
        yk1=0.6*yk+0.05*uk_3
        ykstack[i-1]=yk1
    
    
        utest_start=ustack[0,1:lstm_length,0]
    
        ytest_start=yk1
        yreftest_start=yreftest[i]
        NNtest_start=np.column_stack((bias,ytest_start))
        NNtest_start=np.column_stack((NNtest_start,yreftest_start))
    return ykstack        