from re import A
import numpy as np
import matplotlib.pyplot as plt
from iris_classification import *

def main():
    path = "/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_first30.data"
    values, labels = get_data(path)

    targets = get_targets(labels)
    sepal_length = reshape_data_into_classes(values[:,0], targets)
    plt.hist(sepal_length, bins = 20)
    plt.legend(np.unique(labels))
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('count')
    plt.draw()



    plt.figure()
    sepal_width = reshape_data_into_classes(values[:,1], targets)
    plt.hist(sepal_width, bins = 20)
    plt.legend(np.unique(labels))
    plt.xlabel('Sepal width [cm]')
    plt.ylabel('count')
    plt.draw()


    plt.figure()
    petal_length = reshape_data_into_classes(values[:,2], targets)
    plt.hist(petal_length, bins = 20)
    plt.legend(np.unique(labels))
    plt.xlabel('Petal length [cm]')
    plt.ylabel('count')
    plt.draw()


    plt.figure()
    petal_width = reshape_data_into_classes(values[:,3], targets)
    plt.hist(petal_width, bins = 20)
    plt.legend(np.unique(labels))
    plt.xlabel('Petal width [cm]')
    plt.ylabel('count')
    plt.show()



def reshape_data_into_classes(values, targets):

    class_separated_data = np.reshape(values, (np.sum(targets == 0), len(np.unique(targets))), order = 'F')
    return class_separated_data


if __name__ == '__main__':
    main()