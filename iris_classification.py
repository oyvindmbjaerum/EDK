import math as m
from multiprocessing import set_forkserver_preload
from socket import sethostname
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv 

def main():
    path = '/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_first30.data'
    values, labels = get_data(path)
    #print(values)
    #print(labels)
    targets = get_targets(labels)

    W = init_random_weights(values, targets)

    print(W)

    g = get_discriminant_vec(W, values)

    print(g.shape)
    print(g)
    




#gets data from file and turns into one matrix for data and one for labels
def get_data(path):
    values = np.loadtxt(path, delimiter=',', usecols=[0,1,2,3])
    labels = np.loadtxt(path, delimiter=',', usecols=[4], dtype = str)
    return values, labels

def get_targets(labels):
    targets = np.zeros((len(labels), 1))
    unique_values = np.unique(labels)
    for i in range(len(labels)):
        #this where function is weird, returns tuple of matches  https://numpy.org/doc/stable/reference/generated/numpy.where.html
        match = np.where(unique_values == labels[i])
        targets[i] = match[0]

    targets = targets.astype('int32')
    return targets


def minimum_square_error(values, targets):
    mse = np.square(np.subtract(values, targets)).mean()
    return mse



def init_random_weights(values, targets):
    num_features = values.shape[1]
    num_classes = len(np.unique(targets))

    W = np.random.randn(num_classes, num_features + 1)#last column is for bias
    return W


#discriminant function g_i for each class (i =  1, ..., C)
#g_i = W_i * x + w_io
# on matrix form
# g = [W * w_0][x^T *1]^T


def get_discriminant_vec(W, values):
    
    X = get_X_from_values(values)
    print(X)
    print(X.shape)
    print(W.shape)

    z = W.dot(X)

    print(z.shape)
    return sigmoid_on_matrix(z)

def get_X_from_values(values):
    X = np.c_[values, np.ones((90, 1))] #this last row is for biases to be multiplied with 1 and not a feature
    X = X.T
    return X


def sigmoid_on_matrix(z):
    return 1 / (1 + np.exp(-z))




#separate training data into 3 matrices, one for each class
def separate_classes(values):
    setosa
    vericolor
    virginica


    return setosa, vericolor, virginica
    


#takes in a np.ndarray and normalizes it to values between 0 and 1
def normalize_data():
    b





#Make weights for the single layer of neurons

#Activation function

#iterate with a step factor alpha

#Print out performance of classifier

#Save weights to text file

#I will do a separate file for testing the classifier



if __name__ == '__main__':
    main()
