import numpy as np
import matplotlib.pyplot as plt
from iris_classification import *


def main():
    weight_path = "/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/trainedonlast30.txt"
    testing_data_path = "/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/testing_first20.data.txt"

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