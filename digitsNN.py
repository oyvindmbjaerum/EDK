import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
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
    start_time = time.time()
    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")

    chunk_size = 100
    chunked_test_values = split_test_set_into_chunks(mat_file["testv"], chunk_size)

    pred_mat, index_mat = calc_nearest_neighbour_for_chunk(chunked_test_values[0], mat_file["trainv"], mat_file["trainlab"])
    
    conf_mat = get_conf_matrix(pred_mat, mat_file["testlab"][:chunk_size, :])
    print(conf_mat)
    error_rate = get_error_rate(conf_mat)
    print(error_rate)
    
    #wrongly classified images
    plot_test_sample_and_nearest_neighbour(chunked_test_values[0][1], mat_file["trainv"][index_mat[1]],mat_file["trainlab"][index_mat[1]], mat_file["testlab"][1])
    plot_test_sample_and_nearest_neighbour(chunked_test_values[0][8], mat_file["trainv"][index_mat[8]], mat_file["trainlab"][index_mat[8]], mat_file["testlab"][8])

    #correctly classified images
    plot_test_sample_and_nearest_neighbour(chunked_test_values[0][0], mat_file["trainv"][index_mat[0]], mat_file["trainlab"][index_mat[0]], mat_file["testlab"][0])
    plot_test_sample_and_nearest_neighbour(chunked_test_values[0][3], mat_file["trainv"][index_mat[3]], mat_file["trainlab"][index_mat[3]], mat_file["testlab"][3])
    
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


def plot_test_sample_and_nearest_neighbour(test_value, nearest_neighbour, nn_label, testlabel):
    image = np.reshape(test_value, (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    #prediction = "".join(["Label: ", str(testlabel)])
    #plt.title(prediction)
    plt.draw()

    plt.figure()
    nearest_image = nearest_neighbour
    nearest_image = np.reshape(nearest_image, (28, 28))
    #label = "".join(["Nearest neighbour: ", str(nn_label)])
    #plt.title(label)
    plt.imshow(nearest_image, cmap='gray_r')
    plt.draw()

def calc_nearest_neighbour_for_chunk(chunk, trainv, trainlab):
    pred_mat = np.zeros( chunk.shape[0], dtype = 'int32')
    index_mat =  np.zeros( chunk.shape[0], dtype = 'int32')
    for j in range(0, chunk.shape[0]):
        distance_vector = calc_euclidean_distances(chunk[j], trainv)
        pred, index = find_nearest_neighbour(distance_vector, trainlab)
        #print(pred, index)
        pred_mat[j] = pred
        index_mat[j] = index

    return pred_mat, index_mat



if __name__ == '__main__':
    main()