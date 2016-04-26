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
from keras.utils.visualize_util import plot, to_graph
import copy

# Function to save the Keras model
# Inputs:
#   1. model: model to save
#   2. file_name: desired name to save the model under, do not include the extension
#   3. weight_file_name: desired name to save the weights under, do not include the extension
# Outputs:
#   None
def save_model(model, file_name, weight_file_name):
    # save model
    from keras.models import model_from_json
    json_string = model.to_json()
    open(file_name + '.json', 'w').write(json_string)
    model.save_weights(weight_file_name + '_weights.h5')
    return

# Function to save the Keras model
# Inputs:
#   1. file_name: path to the model file, do not include the .json extension
#   2. weight_file_name: path to the weights file, do not include the .h5 extension
# Outputs:
#   1. model: the model
def load_model(file_name, weight_file_name):
    # load model
    from keras.models import model_from_json
    start_time = time.time()
    model = model_from_json(open(file_name + '.json').read())
    model.load_weights(weight_file_name + '.h5')
    print("Load Time : %s seconds ---" % (time.time() - start_time))
    return model

# Function to extract data file
# Inputs:
#   1. path: path to the file that should be extracted
#   2. sequence_length: sequence length of the LSTM that is desired
# Outputs:
#   1. result: 3-D matrix containing the training or testing data
def extract_data(path, sequence_length):
    with open(path) as f:
        data = csv.reader(f, delimiter=";")
        power = []  # we first parse the data file and put each line in a list
        # nb_of_values = 0
        for line in data:
            try:
                feat_str = line[0].split("\t")  # we expect 2 values, U and Y, for each line separated by a tab
                features = [] # need to convert all features from string into floats
                for feat in feat_str:
                    features.append(float(feat))
                power.append(features)
                # nb_of_values += 1
            except ValueError:
                pass

    print "Data loaded from csv. Formatting..."
    result = []
    # Now we use a sliding window to make training vectors of length sequence_length
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)
    return result

# Function  to import and prepare the data
# Inputs:
#   1. path_to_dataset: path to the training file
#   2. sequence_length: sequence length of the LSTM that is desired
#   3. ratio: the percentage of the training data to be used in training
#   4. path_to_test: path to the test file. The default is to t est on the last 1-ratio part
#      of the training file. You can specify the path to a separate training data file
# Outputs:
#   1. training and test data
def get_data(path_to_dataset="./data/manySines.txt", sequence_length=13, ratio=0.995, path_to_test = 0):

    result = extract_data(path_to_dataset, sequence_length)  # training data

    # offset by the mean
    result_mean = np.mean(result, 0) # result is #trainingpoints by #sequence_length by 2, sum over #trainingpoints
    result_mean = np.mean(result_mean, 0) # sum over sequence_length
    result -= result_mean
    print "Mean Shift : ", result_mean
    print "Data Shape: ", result.shape
    print "Data Example: First Row \n"
    print result[0]

    row = round(ratio * result.shape[0])  # last row to train on

    # whether to test on last part of training data or a separate test data file
    if path_to_test == 0:
        # we don't want to shuffle the test data so we split it here
        X_test = copy.copy(result[row:, :-1])
        y_test = copy.copy(result[row:, -1])
    else:
        test_data = extract_data(path_to_test, sequence_length)
        test_mean = np.mean(test_data, 0) # result is #trainingpoints by #sequence_length by 2, sum over #trainingpoints
        test_mean = np.mean(test_mean, 0) # sum over sequence_length
        test_data -= test_mean
        print "Test Mean Shift : ", result_mean
        print "Data Shape: ", result.shape

        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]

    # shuffle the training data
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    # Get rid of the Y's for targets, keep only the U's
    y_train = y_train[:, 0]

    return [X_train, y_train, X_test, y_test]

# Function to design and construct the model
# Inputs:
#   1. data_dim: Dimension of training data, in this case 2 because U and Y
#   2. timesteps: length of LSTM
# Outputs:
#   1. Keras model
def design_model(data_dim, timesteps):
    model = Sequential()
    output_hidden_size = [10, 100]
    drop_out_rate = [0.2, 0.2]
    model.add(LSTM(output_hidden_size[0], return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(Dropout(drop_out_rate[0]))  # return_sequences=True means output cell state C at each LSTM sequence
    model.add(LSTM(output_hidden_size[1], return_sequences=False))
    model.add(Dropout(drop_out_rate[1]))  # return_sequence=False means output only last cell state C in LSTM sequence

    # final dense layer
    model.add(Dense(1, activation='linear'))

    # output the model to a PNG file for visualization
    print "Outputting model graph to model.png"
    graph = to_graph(model, show_shape=True)
    graph.write_png("model.png")

    return model


# ------------------------------------- Main Loop --------------------------------------------
# Get the data
[X_train, y_train, X_test, y_test]=get_data("./data/manySinesWithRef.txt", 13, 1, "./data/testFile.txt")
X_train = X_train[:, :, 0:2]
X_test = X_test[:, :, 0:2]

# define the input sizes for the LSTM
data_dim = X_train.shape[2]
timesteps = 40

# construct and compile the model
model = design_model(data_dim, timesteps)
start_time = time.time()
print "Compiling Model ..."
model.compile(loss="mse", optimizer="rmsprop")
print("Compile Time : %s seconds --- \n" % (time.time() - start_time))

# train the model
# model.fit(X_train, y_train, batch_size=512, nb_epoch=1, validation_split=0.05)
# training parameters
my_batch_size = 512
my_epoch = 5
start_time = time.time()
model.fit(X_train, y_train, batch_size=my_batch_size, nb_epoch=my_epoch)
print("Training Time : %s seconds --- \n" % (time.time() - start_time))

# test the model
U_hat = model.predict(X_test,verbose=1)
U_hat = U_hat.reshape((len(U_hat)))
loss_and_metrics = model.evaluate(X_test, y_test[:, 0])

# plot the predicted versus the actual U values
import matplotlib.pyplot as plt
toPlot = np.column_stack((U_hat, y_test[:, 0]))
plt.plot(toPlot)
plt.show()