from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from NNclassifier import *
from clustering_NN import *

def main():
    data_mat_file = loadmat("data_all.mat")

    num_neighbours = 1
    print("*----------------------------------------------*")
    print("NEAREST NEIGHBOUR CLASSIFIER")
    predicted_labels = classify_by_kNN(data_mat_file["trainv"], data_mat_file["trainlab"], data_mat_file["testlab"], data_mat_file["testv"], num_neighbours)
    print("*----------------------------------------------*\n")

    plot_wrongly_classified_digits(data_mat_file["testlab"], predicted_labels, data_mat_file["testv"])
    plot_correctly_classified_digits(data_mat_file["testlab"], predicted_labels, data_mat_file["testv"])
    
    num_clusters = 64
    clustered_training_values, clustered_labels = cluster_training_set(data_mat_file["trainv"], data_mat_file["trainlab"], num_clusters)
    print("*----------------------------------------------*")
    print("NEAREST NEIGHBOUR CLASSIFIER WITH CLUSTERING")
    classify_by_kNN(clustered_training_values, clustered_labels, data_mat_file["testlab"], data_mat_file["testv"], num_neighbours)
    print("*----------------------------------------------*\n")
    
    num_neighbours = 7
    print("*----------------------------------------------*")
    print("7 NEAREST NEIGHBOUR CLASSIFIER")
    classify_by_kNN(data_mat_file["trainv"], data_mat_file["trainlab"], data_mat_file["testlab"], data_mat_file["testv"], num_neighbours)
    print("*----------------------------------------------*\n")
    plt.show()

def classify_by_kNN(trainv, trainlab, testlab, testv, num_neighbours):
    pred_start_time = time.time()
    classifier = KNeighborsClassifier(n_neighbors = num_neighbours, metric = 'euclidean')
    classifier.fit(trainv, trainlab.ravel())
    pred_digits = classifier.predict(testv)
    conf_mat = confusion_matrix(testlab, pred_digits)
    print(conf_mat)   
    print(classification_report(testlab, pred_digits))
 
    print("ERROR RATE: ", get_error_rate(conf_mat))
    print("PREDICTION TIME: ", (time.time() - pred_start_time))

    return pred_digits

def cluster_training_set(trainv, trainlab, num_clusters):
    classes = separate_training_values(trainv, trainlab)
    clustered_training_set = get_clustered_training_values(classes, num_clusters)
    clustered_labels = generate_clustered_training_labels(clustered_training_set, np.unique(trainlab))

    return clustered_training_set, clustered_labels

if __name__ == '__main__':
    main()