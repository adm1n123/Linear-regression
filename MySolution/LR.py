import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        return None  # raise NotImplementedError

    def __call__(self, features, is_train=False):
        normalized = (features-features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
        # normalized = (features-features.mean()) / features.std()
        return normalized  # raise NotImplementedError


def get_features(csv_path, is_train=False, scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''
    if is_train:
        cols_name = pd.read_csv(csv_path, nrows=1).columns.tolist()  # get column names in list
        features = pd.read_csv(csv_path, usecols=cols_name[:-1])  # get all except last col, last col is target.
    else:
        features = pd.read_csv(csv_path)

    features = features.to_numpy()
    # features = np.delete(features, 52, 1)
    # rows, cols = features.shape
    # bias = np.zeros((rows, 1))
    # bias.fill(1)
    # features = np.append(features, bias, 1)

    features[:, 52] = 1 # using column 52 as bias
    # numerator_feature = features - features.min(axis=0)
    # denominator_vector = features.max(axis=0) - features.min(axis=0)
    # features = numerator_feature/denominator_vector
    return features

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''


def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    target = pd.read_csv(csv_path).iloc[:, -1]
    return target.to_numpy()


def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''

    # w = (xTx + C*n I)^-1 . xTy
    x = np.copy(feature_matrix)
    y = np.copy(targets)
    xt = x.transpose()
    xtx = xt.dot(x)
    row, col = xtx.shape
    n = y.size
    diagonal = np.zeros((row, row), dtype=float)
    np.fill_diagonal(diagonal, n * C)
    xtx_inverse = np.linalg.inv(np.add(xtx, diagonal))
    xty = xt.dot(y)
    weights = xtx_inverse.dot(xty)
    return weights

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''


def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    prediction = feature_matrix.dot(weights)
    return prediction

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''


def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    prediction = get_predictions(feature_matrix, weights)
    difference = np.subtract(prediction, targets)
    squared_error = np.square(difference)
    mse = squared_error.mean()
    return mse

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''


def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    squared_weights = np.square(weights)
    return np.sum(squared_weights)
    '''
    Arguments
    weights: numpy array of shape n x 1
    '''


def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''
    mse = mse_loss(feature_matrix, weights, targets)
    l2_norm = l2_regularizer(weights)
    return mse + C * l2_norm
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''


def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    # gradient = -2/n . xT(y-xw) + 2C * w
    n = targets.size
    x = np.copy(feature_matrix)
    xt = x.transpose()
    y = np.copy(targets)
    w = np.copy(weights)
    gradient = xt.dot(np.subtract(y, x.dot(w))) * (-2.0/n)
    gradient = np.add(gradient, w * (2*C))
    return gradient

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''


def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''
    row, col = feature_matrix.shape

    index = np.random.randint(0, row-batch_size)
    rand_feature = feature_matrix[index:index+batch_size, :]
    rand_target = targets[index:index+batch_size]
    return rand_feature, rand_target

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''


def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    weights = np.random.rand(n)
    return weights

    '''
    Arguments
    n: int
    '''


def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''
    updated_weights = np.subtract(weights, gradients * lr)
    return updated_weights

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''


def early_stopping(loss, n, idx):
    # allowed to modify argument list as per your need
    # return True or False

    if loss[(idx+1)%n]+loss[(idx+2)%n] > loss[idx]+loss[idx-1]:
        return False
    else:
        return True


def do_gradient_descent(train_feature_matrix,
                        train_targets,
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows -- 
    '''

    weights = initialize_weights(train_feature_matrix.shape[1])
    # weights = analytical_solution(train_features, train_targets, C=1e-9)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    loss_len = 10000
    loss_idx = 0
    patience = 50
    loss = [1e100] * loss_len

    print("step {} \t dev loss: {} \t train loss: {}".format(0, dev_loss, train_loss))
    for step in range(1, max_steps + 1):

        # sample a batch of features and gradients
        features, targets = sample_random_batch(train_feature_matrix, train_targets, batch_size)

        # compute gradients
        gradients = compute_gradients(features, weights, targets, C)

        # update weights
        weights = update_weights(weights, gradients, lr)

        if step % eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step, dev_loss, train_loss))

            # implementing early stopping
            loss[loss_idx] = dev_loss
            loss_idx = (loss_idx+1) % loss_len
            if early_stopping(loss, loss_len, loss_idx):
                patience -= 1;
                if patience == 0:
                    print("Stopping Early")
                    break


        '''
        implement early stopping etc. to improve performance.
        '''

    return weights


def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss = mse_loss(feature_matrix, weights, targets)
    return loss


def output_csv(prediction):
    prediction = prediction
    df = pd.DataFrame(data=prediction)
    df.to_csv("data/prediction.csv")


if __name__ == '__main__':
    scaler = Scaler()  # use of scaler is optional
    train_features, train_targets = get_features('data/train.csv', True, scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv', True, scaler), get_targets('data/dev.csv')
    test_features = get_features('data/test.csv')

    #train_features = scaler.__call__(train_features)
    #dev_features = scaler.__call__(dev_features)
    #output_csv(dev_features)
    train_features = np.concatenate((train_features, dev_features), axis=0)  # combine train and dev data
    train_targets = np.concatenate((train_targets, dev_targets), axis=0)  # combine train and dev data
    a_solution = analytical_solution(train_features, train_targets, C=1e-10)
    test_prediction = get_predictions(test_features, a_solution)
    output_csv(test_prediction)

    print('evaluating analytical_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, a_solution)
    train_loss = do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features,
                                                train_targets,
                                                dev_features,
                                                dev_targets,
                                                lr=1e-15,
                                                C=1e-9,
                                                batch_size=32,
                                                max_steps=2000000,
                                                eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
