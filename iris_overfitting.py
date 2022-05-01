import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from iris_training import *

def main():
    all_features = [0, 1, 2, 3]
    path = 'training_first30.data'
    data, labels = get_data(path, all_features)
    targets = get_target_mat(labels)
    W = init_random_weights(data, labels)
    alpha = 0.25
    norm_data = normalize_data_mat(data)
    X = get_X_from_values(norm_data)
    iterations = 100000

    testing_path = "test_last20.data"

    test_data, test_labels = get_data(testing_path, all_features)
    test_targets = get_target_mat(test_labels)
    mse = np.zeros((iterations,))
    test_mse = np.zeros((iterations, ))
    X_test = get_X_from_values(normalize_data_mat(test_data))

    for i in range(0, iterations):
        G = get_discriminant_vec(W, X)
        mse_grad = calculate_mse_gradient(G, targets, X)
        W = W - alpha * mse_grad
        mse[i] = calculate_mse(G, targets)

        G_test = get_discriminant_vec(W, X_test)
        test_mse[i] = calculate_mse(G_test, test_targets)

    plt.figure()
    plt.plot(mse)
    plt.plot(test_mse)
    plt.show()

if __name__ == "__main__":
    main()