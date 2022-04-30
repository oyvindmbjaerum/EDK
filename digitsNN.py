import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from iris_testing import *
from scipy.spatial.distance import cdist

#From the .mat file we want only the variables, none of the metadata
#vec_size  -> number of features
#trainv    -> values of training set
#num_train -> number of training samples
#num_test  -> number of test samples
#testv     -> values of test set
#trainlab  -> labels of training set
#testlab   -> labels of test set

def main():
    start_time = time.time()
    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")

    chunk_size = 1000
    chunked_test_values = split_test_set_into_chunks(mat_file["testv"], chunk_size)
    predicted_labels, index_vec = calc_nearest_neighbour_for_chunk(chunked_test_values[0], mat_file["trainv"], mat_file["trainlab"])
    
    conf_mat = get_conf_matrix(predicted_labels, mat_file["testlab"][:chunk_size, :])
    print(conf_mat)
    error_rate = get_error_rate(conf_mat)
    print(error_rate)
    
    plot_correctly_classified_digits_with_neighbours(mat_file["testlab"], predicted_labels, mat_file["testv"], mat_file["trainv"], index_vec)
    plot_wrongly_classified_digits_with_neighbours(mat_file["testlab"], predicted_labels, mat_file["testv"], mat_file["trainv"], index_vec)

    print("EXECUTION TIME: ", (time.time() - start_time))
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

def calc_nearest_neighbour_for_chunk(chunk, trainv, trainlab):

    print("Started calculating dist matrix")
    dist_time = time.time()
    dist_mat = cdist(trainv, chunk) #this should make a dist matrix with shape (60000, 10000) 

    print("Distance calculation time: ", (time.time() - dist_time))
    index_vec = np.argmin(dist_mat, axis = 0)    #which we will need to find min value in each column to get nearest neighbour
    predicted_labels = trainlab[index_vec]
    return predicted_labels, index_vec

def plot_wrongly_classified_digits_with_neighbours(testlab, predlab, testv, trainv, index_mat):
    index_wrong_class = []
    for i in range(0, len(predlab)):
        if testlab[i] != predlab[i]:
            index_wrong_class.append(i)

    plot_image(testv, index_wrong_class[0])
    plot_image(trainv, index_mat[index_wrong_class[0]])

    plot_image(testv, index_wrong_class[1])
    plot_image(trainv, index_mat[index_wrong_class[1]])

def plot_correctly_classified_digits_with_neighbours(testlab, predlab, testv, trainv, index_mat):
    index_correct_class = []
    for i in range(0, len(predlab)):
        if testlab[i] == predlab[i]:
            index_correct_class.append(i)

    plot_image(testv, index_correct_class[0])
    plot_image(trainv, index_mat[index_correct_class[0]])

    plot_image(testv, index_correct_class[1])
    plot_image(trainv, index_mat[index_correct_class[1]])

def plot_image(value, index):
    image = np.reshape(value[index], (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    plt.draw()

if __name__ == '__main__':
    main()