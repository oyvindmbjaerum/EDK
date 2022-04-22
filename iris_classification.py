from gc import get_referents
import math as m
import numpy as np
#import scipy
import matplotlib.pyplot as plt
#import csv 
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

def main():
    path = '/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_first30.data'
    data, labels = get_data(path)
    targets = get_target_mat(labels)
    W = init_random_weights(data, labels)
    alpha = 0.75
    norm_data = normalize_data_mat(data)
    X = get_X_from_values(norm_data)
    X_init = get_X_from_values(data)
    #print(X)
    #print(X.shape)

    for i in range(0, 100000):
        G = get_discriminant_vec(W, X)
        mse_grad = calculate_mse_gradient(G, targets, X)

        W = W - alpha * mse_grad

        #print("iteration")
        #print(i)
        #print(W)

   
    predicted_values = get_predicted_values(W, X)
    true_values = get_targets(labels)
    conf_matrix = get_conf_matrix(predicted_values, true_values)

    error_rate = get_error_rate(conf_matrix)
    print(error_rate)




 





#gets data from file and turns into one matrix for data and one for labels
def get_data(path):
    values = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2,3])
    labels = np.loadtxt(path, delimiter=',', usecols=[4], dtype = str)
    return values, labels

#this gives a matrix where true class is 1 all other values is 0
def get_target_mat(labels):
    target_vectors = preprocessing.LabelBinarizer().fit_transform(labels)
    return target_vectors

#this function gives a vector where class target is 0, 1, 2 etc.
def get_targets(labels):
        uniq = np.unique(labels)
        targets = np.zeros(labels.shape, dtype = 'int32')

        for i in range(len(targets)):
            if labels[i] == uniq[0]:
                targets[i] = 0
            if labels[i] == uniq[1]:
                targets[i] = 1
            if labels[i] == uniq[2]:
                targets[i] = 2
        
        return targets
        
def normalize_data_mat(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

def minimum_square_error(values, targets):
    mse = np.square(np.subtract(values, targets)).mean()
    return mse

def init_random_weights(values, labels):
    num_features = values.shape[1]
    num_classes = len(np.unique(labels))
    W = np.random.randn(num_classes, num_features + 1)#last column is for bias
    return W

def get_discriminant_vec(W, X):
    z = W.dot(X.T)
    return sigmoid_on_matrix(z)

def get_X_from_values(values):
    X = np.c_[values, np.ones((90, 1))] #this last row is for biases to be multiplied with 1 and not a feature
    return X

def sigmoid_on_matrix(z):
    return 1 / (1 + np.exp(-z))

def calculate_mse_gradient(G, targets, X):
    new_matrix = (G - targets.T) * G * (1-G)

    W_new = new_matrix @ X
    return W_new

def get_predicted_values(W, X):
    G_final = W @ X.T
    predicted_values = np.argmax(G_final, axis=0)
    return predicted_values



def get_conf_matrix(pred, true):
    print(pred)
    print((np.unique(pred)))
    conf_matrix = np.zeros((len(np.unique(pred)), len(np.unique(pred))))
    conf_matrix = confusion_matrix(true, pred)
    print(conf_matrix)
    return conf_matrix

def get_error_rate(conf_matrix):
    dia = np.diag_indices(len(conf_matrix[0]))
    dia_sum = sum(conf_matrix[dia])
    off_dia_sum = np.sum(conf_matrix) - dia_sum
    return off_dia_sum/np.sum(conf_matrix)




#Make weights for the single layer of neurons

#Activation function

#iterate with a step factor alpha

#Print out performance of classifier

#Save weights to text file

#I will do a separate file for testing the classifier



if __name__ == '__main__':
    main()
