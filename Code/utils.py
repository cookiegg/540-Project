import numpy as np
import numpy as np
import time
import csv
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
    print "Loading Model ... \n"
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
#      X_train is the data for the LSTM, NN_train is the data for the neural network
def get_data(path_to_dataset="./data/manySines.txt", sequence_length=13, ratio=0.995, path_to_test = 0):

    result = extract_data(path_to_dataset, sequence_length)  # training data

    # offset by the mean
    # result_mean = np.mean(result, 0) # result is #trainingpoints by #sequence_length by 2, sum over #trainingpoints
    # result_mean = np.mean(result_mean, 0) # sum over sequence_length
    # result -= result_mean
    # print "Mean Shift : ", result_mean
    print "Data Shape: ", result.shape
    print "Data Example: First Row \n"
    print result[0]

    row = round(ratio * result.shape[0])  # last row to train on

    # whether to test on last part of training data or a separate test data file
    if path_to_test == 0:
        # we don't want to shuffle the test data so we split it here
        X_test = copy.copy(result[row:, :-1])
        y_test = copy.copy(result[row:, -1])
        NN_test = X_test[:, X_test.shape[1]-1, [1, 2]]
        NN_test = np.column_stack((np.ones((NN_test.shape[0], 1)), NN_test))
    else:
        test_data = extract_data(path_to_test, sequence_length)
        # # Mean shift
        # test_mean = np.mean(test_data, 0) # result is #trainingpoints by #sequence_length by 2, sum over #trainingpoints
        # test_mean = np.mean(test_mean, 0) # sum over sequence_length
        # test_data -= test_mean
        # print "Test Mean Shift : ", test_mean
        print "Data Shape: ", test_data.shape

        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        NN_test = X_test[:, X_test.shape[1]-1, [1, 2]]
        NN_test = np.column_stack((np.ones((NN_test.shape[0], 1)), NN_test))

    # shuffle the training data
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    NN_train = X_train[:, X_train.shape[1]-1, [1, 2]]
    NN_train = np.column_stack((np.ones((NN_train.shape[0], 1)), NN_train))  # add bias

    # Get rid of the Y's for targets, keep only the U's
    y_train = y_train[:, 0]

    return [X_train, y_train, X_test, y_test, NN_train, NN_test]