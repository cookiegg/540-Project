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

def data_power_consumption(path_to_dataset="../data/BigsinData.txt",
                           sequence_length=13,
                           ratio=0.995):

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                u, y = line[0].split("\t")
                power.append([float(u), float(y)])
                nb_of_values += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            # if nb_of_values >= max_values:
            #     break

    print "Data loaded from csv. Formatting..."
    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)

    # we don't want to shuffle the test data so we split it here
    row = round(ratio * result.shape[0])
    X_test = copy.copy(result[row:, :-1])
    control_unit_test = X_test[:,:,0]
    state_val_test = X_test[:,:,1]
    y_test = copy.copy(result[row:, -1])

    # offset by the mean
    result_mean = np.mean(result, 0)
    # need to sum over the 10/12
    result_mean = np.mean(result_mean, 0)
    result -= result_mean
    print "Shift : ", result_mean
    print "Data  : ", result.shape
    print "sth:", result[0]

    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    control_unit_train = X_train[:,:,0]
    state_val_train = X_train[:,:,1]
    y_train = train[:, -1]

    # Get rid of the Y's for targets, keep only the U's
    y_train = y_train.tolist()
    for row in y_train:
        del row[1]
    y_train = np.asarray(y_train)
    y_train = np.reshape(y_train, (y_train.shape[0]))

    control_unit_train = np.reshape(control_unit_train, (control_unit_train.shape[0], control_unit_train.shape[1], 1))
    state_val_train = np.reshape(state_val_train, (state_val_train.shape[0], state_val_train.shape[1], 1))
    control_unit_test = np.reshape(control_unit_test, (control_unit_test.shape[0], control_unit_test.shape[1], 1))
    state_val_test = np.reshape(state_val_test, (state_val_test.shape[0], state_val_test.shape[1], 1))


    return [control_unit_train, state_val_train,y_train, control_unit_test, state_val_test, y_test]

[control_unit_train, state_val_train,y_train, control_unit_test, state_val_test, y_test]=data_power_consumption()


data_dim = 1
timesteps = 13

# expected input data shape: (batch_size, timesteps, data_dim)
model_A = Sequential()
model_B = Sequential()
model_Combine = Sequential()


model_A.add(LSTM(13, input_shape=(timesteps, data_dim)))
model_B.add(LSTM(13, input_shape=(timesteps, data_dim)))
model_Combine.add(Merge([model_A, model_B], mode='sum'))

model_Combine.add(Dense(1, activation='linear'))

model_Combine.compile(loss="mse", optimizer="rmsprop")

model_Combine.fit([control_unit_train, state_val_train], y_train, batch_size=512, nb_epoch=5)

# predicted = model.predict(X_test)
# predicted = np.reshape(predicted, (predicted.size,))



U_hat = model_Combine.predict([control_unit_test,state_val_test],verbose=1)
U_hat = U_hat.reshape((len(U_hat)))
import matplotlib.pyplot as plt
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()

# output the model
from keras.utils.visualize_util import plot, to_graph
graph = to_graph(model_Combine, show_shape=True)
graph.write_png("modelsteven13.png")

#predict function with 1 values
#utest=control_unit_test[1:2,:,:]
#ytest=state_val_test[1:2,:,:]
#
#
#uhat1=model_Combine.predict([utest,ytest],verbose=1)    
#
##unew = np.zeros((1,13,1))
#
#unew=np.append(utest,uhat1)
#unew=unew.reshape(1,len(unew),1)
##unew=[unew]
#
#uk_3=unew[0,8,0]
#yk=ytest[0,11,0]
#
#yk1=0.6*yk+0.05*uk_3

#predict function with 1 values ends

#iterative version:
utest=control_unit_test[1:2,:,:]
ytest=state_val_test[1:2,:,:]
ykstack=np.zeros(shape=(100,1))
for i in range(1,101):
    print i
    uhat=model_Combine.predict([utest,ytest],verbose=1)
    
    ustack=np.append(utest,uhat)
    ustack=ustack.reshape(1,len(ustack),1)
    
    uk_3=ustack[0,8,0]
    yk=ytest[0,11,0]
    
    yk1=0.6*yk+0.05*uk_3
    ykstack[i-1]=yk1
    ystack=np.append(ytest,yk1)
    ystack=ystack.reshape(1,len(ystack),1)
    
    utest=ustack[0,1:13,0]
    utest=utest.reshape(1,len(utest),1)
    ytest=ystack[0,1:13,0]
    ytest=ytest.reshape(1,len(ytest),1)
    
    
    
    
    
    






    
