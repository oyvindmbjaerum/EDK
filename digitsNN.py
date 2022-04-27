import chunk
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.io import loadmat
from scipy.spatial import distance


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

    chunk_size = 1000
    num_chunks = int(len(mat_file["testv"])/chunk_size)
    chunked_test_values = split_test_set_into_chunks(mat_file["testv"], num_chunks)
    
    print( mat_file["testv"][3088][96])
    print(chunked_test_values[3][88][96])

    distance_vector = calc_euclidean_distances(chunked_test_values[0][0], mat_file["trainv"])
    print(distance_vector)



#split the test set into num_chunks layered 3d array
def split_test_set_into_chunks(testv, num_chunks):
    mat_size = (num_chunks, int(testv.shape[0]/num_chunks), testv.shape[1]) #layers, rows, columns
    chunked_test_values = testv.reshape(mat_size)
    return chunked_test_values

def calc_euclidean_distances(test_value, trainingv):
    dist_vector = np.linalg.norm(trainingv - test_value, axis = 1)
    return dist_vector


if __name__ == '__main__':
    main()