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

def data_power_consumption(path_to_dataset="./data/BigsinData.txt",
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

# output the model
# from keras.utils.visualize_util import plot, to_graph
# graph = to_graph(model, show_shape=True)
# graph.write_png("model.png")

U_hat = model_Combine.predict([control_unit_test,state_val_test],verbose=1)
U_hat = U_hat.reshape((len(U_hat)))
import matplotlib.pyplot as plt
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()