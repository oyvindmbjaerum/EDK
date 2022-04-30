from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

def main():
    start_time = time.time()
    mat_file = loadmat("/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/data_all.mat")

    pred_start_time = time.time()
    classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'euclidean')
    classifier.fit(mat_file["trainv"], mat_file["trainlab"])
    pred_digits = classifier.predict(mat_file["testv"])
    conf_mat = confusion_matrix(mat_file["testlab"], pred_digits)
    print(conf_mat)   
    print(classification_report(mat_file["testlab"], pred_digits))
    
    print("ERROR RATE: ", get_error_rate(conf_mat))
    print("PREDICTION TIME: ", (time.time() - pred_start_time))

    plot_wrongly_classified_digits(mat_file["testlab"], pred_digits, mat_file["testv"])
    plot_correctly_classified_digits(mat_file["testlab"], pred_digits, mat_file["testv"])
    print("TOTAL TIME: ", (time.time() - start_time))

    plt.show()


def get_conf_matrix(pred, true):
    conf_matrix = np.zeros((len(np.unique(pred)), len(np.unique(pred))))
    conf_matrix = confusion_matrix(true, pred)
    return conf_matrix

def get_error_rate(conf_matrix):
    diagonal_indices = np.diag_indices(len(conf_matrix[0]))
    diagonal_sum = sum(conf_matrix[diagonal_indices]) #correctly classified
    off_diagonal_sum = np.sum(conf_matrix) - diagonal_sum #wrongly classified
    return off_diagonal_sum/np.sum(conf_matrix)

def plot_wrongly_classified_digits(testlab, predlab, testv):
    index_wrong_class = []
    for i in range(0, len(predlab)):
        if testlab[i] != predlab[i]:
            index_wrong_class.append(i)


    image = np.reshape(testv[index_wrong_class[0]], (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    prediction = "".join(["Prediction: ", str(predlab[index_wrong_class[0]]), "\nLabel", str(testlab[index_wrong_class[0]])])
    plt.title(prediction)
    plt.draw()

    image = np.reshape(testv[index_wrong_class[1]], (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    prediction = "".join(["Prediction: ", str(predlab[index_wrong_class[1]]), "\nLabel", str(testlab[index_wrong_class[1]])])
    plt.title(prediction)
    plt.draw()

def plot_correctly_classified_digits(testlab, predlab, testv):
    index_correct_class = []
    for i in range(0, len(predlab)):
        if testlab[i] == predlab[i]:
            index_correct_class.append(i)

    image = np.reshape(testv[index_correct_class[0]], (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    prediction = "".join(["Prediction: ", str(predlab[index_correct_class[0]]), "\nLabel", str(testlab[index_correct_class[0]])])
    plt.title(prediction)
    plt.draw()

    image = np.reshape(testv[index_correct_class[1]], (28, 28))
    plt.figure()
    plt.imshow(image, cmap='gray_r')
    prediction = "".join(["Prediction: ", str(predlab[index_correct_class[1]]), "\nLabel", str(testlab[index_correct_class[1]])])
    plt.title(prediction)
    plt.draw()


if __name__ == '__main__':
    main()