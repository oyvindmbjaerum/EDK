import math as m
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv 

def main():
    path = '/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_first30.data'
    values, labels = get_data(path)
    print(values)
    print(labels)

#gets data from file and turns into one matrix for data and one for labels
def get_data(path):
    values = np.loadtxt(path, delimiter=',', usecols=[0,1,2,3])
    labels = np.loadtxt(path, delimiter=',', usecols=[4], dtype = np.str)
    return values, labels


#separate training data into 3 matrices, one for each class
def separate_classes():
    a


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
