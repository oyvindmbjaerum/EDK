from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from iris_testing import *

def main():
    start_time = time.time()
    chunk_size = 100
    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(mat_file["trainv"], mat_file["trainlab"])
    pred_digits = classifier.predict(mat_file["testv"][:chunk_size, :])
    conf_mat = confusion_matrix(mat_file["testlab"][:chunk_size, :], pred_digits)
    print(conf_mat)   
    print(classification_report(mat_file["testlab"][:chunk_size, :], pred_digits))
 
    print("ERROR RATE: ", get_error_rate(conf_mat))
    print("EXECUTION TIME: ", (time.time() - start_time))





if __name__ == '__main__':
    main()