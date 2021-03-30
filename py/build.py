# Python Build for training, testing and exporting model
import ast
import json
import logging
# Importing Libraries
import os
import pickle
from enum import Enum
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger(__name__)

train_history_pickle_file_suffix = '_train_history.pickle'
test_history_pickle_file_suffix = '_test_history.pickle'

pickle_scalar_file_suffix = '_scalar.pickle'

window_size = 60


def save_sklearn_model(model, model_path, approach='pickle'):
    if approach == 'pickle':
        output = open(model_path, 'wb')
        pickle.dump(model, output)
        output.close()
    elif approach == 'joblib':
        joblib.dump(model, model_path)


def load_sklearn_model(model_path, approach='pickle'):
    model = None
    if approach == 'pickle':
        input_model = open(model_path, 'rb')
        model = pickle.load(input_model)
        input_model.close()
    elif approach == 'joblib':
        model = joblib.load(model_path)
    return model


def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['guid']


def get_feature_label_sklearn(data, default_window_size=60):
    log.debug("window_size: %d" % default_window_size)
    log.debug("data length: %d" % len(data))
    # creating a data structure with window_size time steps
    # for example, given window_size = 60, each sample has 60 values and 1 label
    # X_0 (index 0 to 60), and y_0 (index 60)
    # X_1 (index 1 to 61), and y_1 (index 61)
    # X_2 (index 2 to 62), and y_2 (index 62)
    # ...
    # X_n (index n to (n + 60)), and y_2 (index (n + 60))
    x_train = []
    y_train = []
    for i in range(default_window_size, len(data)):
        x_train.append(data[(i - default_window_size): i])
        y_train.append(data[i])
    return x_train, y_train


def get_feature_label_spark(training_data, default_window_size=60):
    log.debug("window_size: %d" % default_window_size)
    log.debug("training data length: %d" % len(training_data))
    # creating a data structure with window_size time steps
    # for example, given window_size = 60, each sample has 60 values and 1 label
    # X_0 (index 0 to 60), and y_0 (index 60)
    # X_1 (index 1 to 61), and y_1 (index 61)
    # X_2 (index 2 to 62), and y_2 (index 62)
    # ...
    # X_n (index n to (n + 60)), and y_2 (index (n + 60))
    x_train = []
    y_train = []
    for i in range(default_window_size, len(training_data)):
        x_train.append(np.array(training_data[(i - default_window_size): i]).astype(np.float64).tolist())
        y_train.append(np.array(training_data[i]).astype(np.float64).tolist())
    return x_train, y_train


def get_model_base_path():
    return str(Path.cwd() / 'saved_model')

saved_tf_model_directory = get_model_base_path()


def r2_score(y_true, y_pred):
    ss_res = k.sum(k.square(y_true - y_pred))
    ss_tot = k.sum(k.square(y_true - k.mean(y_true)))
    return 1 - ss_res / (ss_tot + k.epsilon())


def get_scaled_data(training_set, file_name):
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    log.debug("training_set_scaled shape:" + str(training_set_scaled.shape))
    log.debug(training_set_scaled)

    pickle_out = open(file_name + pickle_scalar_file_suffix, 'wb')
    pickle.dump(sc, pickle_out)
    pickle_out.close()

    # creating a data structure with window_size time steps
    # for example, given window_size = 60, each sample has 60 values and 1 label
    # X_0 (index 0 to 60), and y_0 (index 60)
    # X_1 (index 1 to 61), and y_1 (index 61)
    # X_2 (index 2 to 62), and y_2 (index 62)
    # ...
    # X_n (index n to (n + 60)), and y_2 (index (n + 60))

    x_train = []
    y_train = []
    for i in range(window_size, training_set_scaled.shape[0] + 1):
        x_train.append(training_set_scaled[i - window_size:i, 0])
        y_train.append(training_set_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def save_plot(local_test, predicted, file_name):
    plt.plot(local_test, color='red', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(file_name + '.jpg')


def get_x_pred(inputs):
    x_pred = []
    for i in range(window_size, inputs.shape[0]):
        x_pred.append(inputs[i - window_size:i, 0])
    x_pred = np.array(x_pred)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    return x_pred


def response(scores=None, prediction=None):
    if scores is None:
        scores = []
    if prediction is None:
        prediction = []
    response_json = {'Eval': scores, 'Pred': prediction}

    return json.dumps(response_json) + "\r\n"
