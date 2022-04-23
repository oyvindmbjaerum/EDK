import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def main():
    path = '/Users/oyvindmasdalbjaerum/SKOLEGREIER/EDK/training_last30.data.txt'
    data, labels = get_data(path)
    targets = get_target_mat(labels)
    W = init_random_weights(data, labels)
    alpha = 0.75
    norm_data = normalize_data_mat(data)
    X = get_X_from_values(norm_data)

    for i in range(0, 100000):
        G = get_discriminant_vec(W, X)
        mse_grad = calculate_mse_gradient(G, targets, X)
        W = W - alpha * mse_grad

    np.savetxt('trainedonlast30.txt', W)

#gets data from file and turns into one matrix for data and one for labels
def get_data(path):
    values = np.loadtxt(path, delimiter=',', usecols=[0, 1, 2,3])
    labels = np.loadtxt(path, delimiter=',', usecols=[4], dtype = str)
    return values, labels

#this gives a matrix where true class is 1 all other values is 0
def get_target_mat(labels):
    target_vectors = preprocessing.LabelBinarizer().fit_transform(labels)
    return target_vectors

#this function gives a vector where class target is 0, 1, 2 etc.
def get_targets(labels):
        uniq = np.unique(labels)
        targets = np.zeros(labels.shape, dtype = 'int32')

        for i in range(len(targets)):
            if labels[i] == uniq[0]:
                targets[i] = 0
            if labels[i] == uniq[1]:
                targets[i] = 1
            if labels[i] == uniq[2]:
                targets[i] = 2
        
        return targets
        
def normalize_data_mat(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

def minimum_square_error(values, targets):
    mse = np.square(np.subtract(values, targets)).mean()
    return mse

def init_random_weights(values, labels):
    num_features = values.shape[1]
    num_classes = len(np.unique(labels))
    W = np.random.randn(num_classes, num_features + 1)#last column is for bias
    return W

def get_discriminant_vec(W, X):
    z = W.dot(X.T)
    return sigmoid_on_matrix(z)

def get_X_from_values(values):
    X = np.c_[values, np.ones((len(values[:,0]), 1))] #this last row is for biases to be multiplied with 1 and not a feature
    return X
#this is our loss function
def sigmoid_on_matrix(z):
    return 1 / (1 + np.exp(-z))

def calculate_mse_gradient(G, targets, X):
    new_matrix = (G - targets.T) * G * (1-G)
    W_new = new_matrix @ X
    return W_new

if __name__ == '__main__':
    main()
