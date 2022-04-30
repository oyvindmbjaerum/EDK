import numpy as np
import matplotlib.pyplot as plt
from iris_training import *
from iris_histograms import *
from iris_testing import *

def main():
    first_30_path = "training_first30.data"
    all_features = [0, 1, 2, 3]


    first_30_values, first_30_labels = get_data(first_30_path, all_features)
   

    plot_histograms_of_features(first_30_values, first_30_labels)
    best_3_features = [0, 2, 3]
    best_2_features = [2, 3]
    best_feature = [3]

    W_first_30 = train_model(first_30_values, first_30_labels)
    print("*----------------------------------------------*")
    print("Model trained with first 30 samples and tested with first 30 samples")
    test_model(W_first_30, first_30_values, first_30_labels)
    print("*----------------------------------------------*\n")

    last_20_path = "test_last20.data"
    last_20_values, last_20_labels = get_data(last_20_path, all_features)
    print("*----------------------------------------------*")
    print("Model trained with first 30 samples and tested with last 20 samples")
    test_model(W_first_30, last_20_values, last_20_labels)
    print("*----------------------------------------------*\n")

    last_30_path = "training_last30.data"
    last_30_values, last_30_labels = get_data(last_30_path, all_features)
    W_last_30 = train_model(last_30_values, last_30_labels)

    print("*----------------------------------------------*")
    print("Model trained with last 30 samples and tested with last 30 samples")
    test_model(W_last_30, last_30_values, last_30_labels)
    print("*----------------------------------------------*\n")

    first_20_path = "test_first20.data"
    first_20_values, first_20_labels = get_data(first_20_path, all_features)

    print("*----------------------------------------------*")
    print("Model trained with last 30 samples and tested with first 20 samples")
    test_model(W_last_30, first_20_values, first_20_labels)
    print("*----------------------------------------------*\n")

    values_3_features, labels_3_features = get_data(first_30_path, best_3_features)
    W_3_features = train_model(values_3_features, labels_3_features)
    values_3_features_test, labels_3_features_test = get_data(last_20_path, best_3_features)
    print("*----------------------------------------------*")
    print("Model trained with 3 features of first 30 samples and tested with last 20 samples")
    test_model(W_3_features, values_3_features_test, labels_3_features_test)
    print("*----------------------------------------------*\n")

    values_2_features, labels_2_features = get_data(first_30_path, best_2_features)
    W_2_features = train_model(values_2_features, labels_2_features)
    values_2_features_test, labels_2_features_test = get_data(last_20_path, best_2_features)
    print("*----------------------------------------------*")
    print("Model trained with 2 features of first 30 samples and tested with last 20 samples")
    test_model(W_2_features, values_2_features_test, labels_2_features_test)
    print("*----------------------------------------------*\n")
    
    values_1_feature, labels_1_feature = get_data(first_30_path, best_feature)
    W_1_feature = train_model(values_1_feature, labels_1_feature)
    values_1_feature_test, labels_1_feature_test = get_data(last_20_path, best_feature)
    print("*----------------------------------------------*")
    print("Model trained with 1 features of first 30 samples and tested with last 20 samples")
    test_model(W_1_feature, values_1_feature_test, labels_1_feature_test)
    print("*----------------------------------------------*\n")

    plt.show()

def plot_histograms_of_features(values, labels):
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

def train_model(training_values, labels):
    targets = get_target_mat(labels)
    W = init_random_weights(training_values, labels)
    alpha = 0.25
    norm_data = normalize_data_mat(training_values)
    X = get_X_from_values(norm_data)
    iterations = 100000

    for i in range(0, iterations):
        G = get_discriminant_vec(W, X)
        mse_grad = calculate_mse_gradient(G, targets, X)
        W = W - alpha * mse_grad

    return W


def test_model(W, testing_data, testing_labels):

    norm_data = normalize_data_mat(testing_data)
    X = get_X_from_values(norm_data)
    pred = get_predicted_values(W, X)
    true = get_targets(testing_labels)
    conf_matrix = get_conf_matrix(pred, true)
    print(conf_matrix)
    error_rate = get_error_rate(conf_matrix)
    print("ERROR RATE: ", error_rate)

if __name__ == '__main__':
    main()