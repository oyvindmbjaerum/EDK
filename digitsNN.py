import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import matplotlib.pyplot as plt
from iris_testing import *


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
    chunked_test_values = split_test_set_into_chunks(mat_file["testv"], chunk_size)
    
    print( mat_file["testv"][3088][96])
    print(chunked_test_values[3][88][96])

    pred_mat = np.zeros( chunked_test_values.shape[1])
    print(pred_mat.shape)

    for j in range(0, chunked_test_values.shape[1]):
        distance_vector = calc_euclidean_distances(chunked_test_values[0][j], mat_file["trainv"])
        pred, index = find_nearest_neighbour(distance_vector, mat_file["trainlab"])
        pred_mat[j] = pred
        print(pred, index)

    
    conf_mat = get_conf_matrix(pred_mat, mat_file["testlab"][:chunk_size, :])
    print(conf_mat)
    error_rate = get_error_rate(conf_mat)
    print(error_rate)
    image = chunked_test_values[0][0]
    image.reshape((28, 28))
    fig = plt.figure()
    plt.imshow(image, cmap='gray_r')
    plt.show()








#split the test set into num_chunks layered 3d array
def split_test_set_into_chunks(testv, chunk_size):
    num_chunks = int(len(testv)/chunk_size)
    mat_size = (num_chunks, int(testv.shape[0]/num_chunks), testv.shape[1]) #layers, rows, columns
    chunked_test_values = testv.reshape(mat_size)
    return chunked_test_values

def calc_euclidean_distances(test_value, trainingv):
    dist_vector = np.linalg.norm(trainingv - test_value, axis = 1)
    return dist_vector

def find_nearest_neighbour(distance_vector, trainlab):
    prediction = trainlab[np.argmin(distance_vector)]
    return prediction, np.argmin(distance_vector)


if __name__ == '__main__':
    main()