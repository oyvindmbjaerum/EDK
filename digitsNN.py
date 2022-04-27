import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.io import loadmat


#From the .mat file we want only the variables, none of the metadata
#vec_size  -> number of features
#trainv    -> values of training set
#num_train -> number of training samples
#num_test  -> number of test samples
#testv     -> values of test set
#trainlab  -> labels of training set
#testlab   -> labels of test set

def main():

    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")
    print(mat_file)

    for key in mat_file:
        print(key, mat_file[key])


if __name__ == '__main__':
    main()