import math as m
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv 




def main():
    get_class_data()


#can also read from the 3 separate files, one for each class, but now i have already separated training and test data
#Get data from CSV file and load it into a np.ndarray
def get_class_data():
    path = '/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_first30.data'
    #data is matrix of N x l + 1, where N is number of samples, l is feature vector length and an extra column for classification in string form
    data = np.loadtxt(path,
    dtype={'names': ('sepal length', 'sepal width', 'petal length', 'petal width', 'label'),
          'formats': (np.float, np.float, np.float, np.float, '|S15')},
    delimiter=',', skiprows=0)

    print(data)

#Workaround:
#values = np.loadtxt('data', delimiter=',', usecols=[0,1,2,3])
#labels = np.loadtxt('data', delimiter=',', usecols=[4])



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
