from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from NNclassifier import *

def main():
    start_time = time.time()
    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")
    num_clusters = 64
    classes = separate_training_values(mat_file["trainv"], mat_file["trainlab"])
    
    clustered_training_values = get_clustered_training_values(classes, num_clusters)
    clustered_trainging_labels = generate_clustered_training_labels(clustered_training_values, np.unique(mat_file["trainlab"]))
  
    pred_time = time.time()
    classifier = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
    classifier.fit(clustered_training_values, clustered_trainging_labels)
    pred_digits = classifier.predict(mat_file["testv"])
    conf_mat = confusion_matrix(mat_file["testlab"], pred_digits)
    print(conf_mat)   
    print(classification_report(mat_file["testlab"], pred_digits))

    print("ERROR RATE: ", get_error_rate(conf_mat))
    print("PREDICTION TIME: ", (time.time() - pred_time))
    print("TOTAL TIME: ", (time.time() - start_time))

def separate_training_values(trainv, trainlab):
    zeros = trainv[(trainlab[:, 0] == 0), :]
    ones = trainv[(trainlab[:, 0] == 1), :]
    twos = trainv[(trainlab[:, 0] == 2), :]
    threes = trainv[(trainlab[:, 0] == 3), :]
    fours = trainv[(trainlab[:, 0] == 4), :]
    fives = trainv[(trainlab[:, 0] == 5), :]
    sixes = trainv[(trainlab[:, 0] == 6), :]
    sevens = trainv[(trainlab[:, 0] == 7), :]
    eights = trainv[(trainlab[:, 0] == 8), :]
    nines = trainv[(trainlab[:, 0] == 9), :]
    return [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]

def get_clustered_training_values(classes, num_clusters):
    kmeans_0 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[0])
    kmeans_1 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[1])
    kmeans_2 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[2])
    kmeans_3 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[3])
    kmeans_4 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[4])
    kmeans_5 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[5])
    kmeans_6 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[6])
    kmeans_7 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[7])
    kmeans_8 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[8])
    kmeans_9 = KMeans(n_clusters= num_clusters, random_state=0).fit(classes[9])

    clustered_training_values = np.concatenate((kmeans_0.cluster_centers_, kmeans_1.cluster_centers_, kmeans_2.cluster_centers_, kmeans_3.cluster_centers_, kmeans_4.cluster_centers_, kmeans_5.cluster_centers_, kmeans_6.cluster_centers_, kmeans_7.cluster_centers_, kmeans_8.cluster_centers_, kmeans_9.cluster_centers_), axis = 0)
    return clustered_training_values

def generate_clustered_training_labels(clustered_training_values, unique_labels):
    clustered_training_labels = np.zeros((clustered_training_values.shape[0],))
    for i in range(0, len(clustered_training_values)):
        clustered_training_labels[i] = unique_labels[i // int(clustered_training_values.shape[0]/len(unique_labels))] 
    return clustered_training_labels

if __name__ == '__main__':
    main()