import numpy as np
import matplotlib.pyplot as plt
from iris_training import *

#File names for testing (working on macos)
#training_first30.data
#test_last20.data
#training_last30.data
#testing_first20.data

#File names for classifiers
#trainedonfirst30.txt
#trainedonlast30.txt
#trainedon3features.txt
#trainedon2features.txt
#trainedon1features.txt

def main():
    weight_path = "/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/trainedonfirst30.txt"
    testing_data_path = "test_last20.data"

    data, labels = get_data(testing_data_path)
    W = get_weight_from_path(weight_path)
    print(W)
    norm_data = normalize_data_mat(data)
    X = get_X_from_values(norm_data)
    pred = get_predicted_values(W, X)
    true = get_targets(labels)
    conf_matrix = get_conf_matrix(pred, true)
    print(conf_matrix)
    error_rate = get_error_rate(conf_matrix)
    print(error_rate)
    



def get_weight_from_path(path):
    W = np.loadtxt(path)
    return W

def get_predicted_values(W, X):
    G_final = W @ X.T
    predicted_values = np.argmax(G_final, axis=0)
    return predicted_values



def get_conf_matrix(pred, true):
    conf_matrix = np.zeros((len(np.unique(pred)), len(np.unique(pred))))
    conf_matrix = confusion_matrix(true, pred)
    return conf_matrix

def get_error_rate(conf_matrix):
    dia = np.diag_indices(len(conf_matrix[0]))
    dia_sum = sum(conf_matrix[dia])
    off_dia_sum = np.sum(conf_matrix) - dia_sum
    return off_dia_sum/np.sum(conf_matrix)

if __name__ == '__main__':
    main()