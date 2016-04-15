from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def data_power_consumption(path_to_dataset="MPCdata.txt",
                           sequence_length=13,
                           ratio=0.05):

    max_values = ratio * 1000

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[0]))
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

    result_mean = result.mean()
    # result -= result_mean
    print "Shift : ", result_mean
    print "Data  : ", result.shape
    print "sth:", result[0]

    row = round(0.25 * result.shape[0])
    train = result[:row, :]
    #before random shuffle, store x for prediction in the end
    # X_plot = train[:100,:-1]

    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[:100, :-1]
    y_test = result[:100, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[0]))
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

    X_plot = result[:100,:-1]
    X_plot = np.reshape(X_plot, (X_plot.shape[0], X_plot.shape[1], 1))

    return [X_train, y_train, X_test, y_test,X_plot]

[X_train, y_train, X_test, y_test,X_plot]=data_power_consumption()


data_dim = 1
timesteps = 49

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(10, return_sequences=True,
               input_shape=(40, 1)))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))  # returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))

# output was 10 for #nb_classes but only 1 now
model.add(Dense(1, activation='linear'))

model.compile(loss="mse", optimizer="rmsprop")

# model.fit(X_train, y_train, batch_size=512, nb_epoch=1, validation_split=0.05)
model.fit(X_train, y_train, batch_size=512, nb_epoch=20)
predicted = model.predict(X_test)
predicted = np.reshape(predicted, (predicted.size,))


haha = model.predict(X_plot,batch_size=10,verbose=1)
haha = haha.reshape((len(haha)))
import matplotlib.pyplot as plt
plt.plot(haha)
plt.show()