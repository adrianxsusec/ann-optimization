import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
import multiprocessing as mp
from numba import jit
import logging
from time import time

matplotlib.use("TkAgg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@jit(nopython=True)
def g(x):
    """ sigmoid function """
    return 1.0 / (1.0 + np.exp(-x))


@jit(nopython=True)
def grad_g(x):
    """ gradient of sigmoid function """
    gx = g(x)
    return gx * (1.0 - gx)


def predict(Theta1, Theta2, X):
    """ Predict labels in a trained three layer classification network.
    Input:
      Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
      Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
      X            matrix of training data      (m x input_layer_size)
    Output:
      prediction   label prediction
    """
    m = np.shape(X)[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)
    prediction = np.argmax(a3, 1).reshape((m, 1))
    return prediction


def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """ reshape theta into Theta1 and Theta2, the weights of our neural network """
    ncut = hidden_layer_size * (input_layer_size + 1)
    Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2


def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function
    Input:
      theta               flattened vector of neural net model parameters
      input_layer_size    size of input layer
      hidden_layer_size   size of hidden layer
      num_labels          number of labels
      X                   matrix of training data
      y                   vector of training labels
      lmbda               regularization term
    Output:
      J                   cost
    """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    # Forward propagation
    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = g(a1 @ Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = g(a2 @ Theta2.T)

    # One-hot encode y
    y_mtx = np.zeros((m, num_labels))
    y_mtx[np.arange(m), y.flatten().astype(int)] = 1

    # Cost computation
    J = np.sum(-y_mtx * np.log(a3) - (1 - y_mtx) * np.log(1 - a3)) / m

    # Regularization
    J += lmbda / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    return J


def _gradient_worker(batch_indices, X, y, Theta1, Theta2, num_labels):
    """Worker function for parallel gradient computation"""
    # Initialize Delta matrices
    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for t in batch_indices:
        # Forward pass
        a1 = X[t, :].reshape((Theta1.shape[1] - 1, 1))
        a1 = np.vstack((1, a1))  # +bias
        z2 = Theta1 @ a1
        a2 = g(z2)
        a2 = np.vstack((1, a2))  # +bias
        a3 = g(Theta2 @ a2)

        # Compute error for layer 3
        y_k = np.zeros((num_labels, 1))
        y_k[y[t, 0].astype(int)] = 1
        delta3 = a3 - y_k
        Delta2 += (delta3 @ a2.T)

        # Compute error for layer 2
        delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)
        Delta1 += (delta2 @ a1.T)

    return Delta1, Delta2


def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """Parallel gradient computation using multiprocessing"""
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    # Split data into batches for parallel processing
    num_processes = mp.cpu_count()
    batch_size = m // num_processes
    batches = [np.arange(i, min(i + batch_size, m))
               for i in range(0, m, batch_size)]

    # Parallel processing
    with mp.Pool(num_processes) as pool:
        try:
            worker = partial(_gradient_worker,
                             X=X,
                             y=y,
                             Theta1=Theta1,
                             Theta2=Theta2,
                             num_labels=num_labels)
            results = pool.map(worker, batches)
        finally:
            pool.close()
            pool.join()

    # Combine results
    Delta1 = sum(r[0] for r in results)
    Delta2 = sum(r[1] for r in results)

    # Compute gradients with regularization
    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m
    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    # Flatten gradients
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad


# Global variables
N_iter = 1
J_min = np.inf
theta_best = []
Js_train = np.array([])
Js_test = np.array([])


def callbackF(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label, theta_k):
    """ Calculate stats per iteration and update plot """
    global N_iter, J_min, theta_best, Js_train, Js_test

    start_time = time()

    # Unflatten theta
    Theta1, Theta2 = reshape(theta_k, input_layer_size, hidden_layer_size, num_labels)

    # Training data stats
    J = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    y_pred = predict(Theta1, Theta2, X)
    accuracy = np.sum(1. * (y_pred == y)) / len(y)
    Js_train = np.append(Js_train, J)

    # Test data stats
    J_test = cost_function(theta_k, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    test_pred = predict(Theta1, Theta2, test)
    accuracy_test = np.sum(1. * (test_pred == test_label)) / len(test_label)
    Js_test = np.append(Js_test, J_test)

    # Print stats with timing
    logger.info(
        f'iter={N_iter:3d} ({time() - start_time:.2f}s): '
        f'Jtrain={J:0.4f} acc={100 * accuracy:0.2f}% | '
        f'Jtest={J_test:0.4f} acc={100 * accuracy_test:0.2f}%'
    )

    N_iter += 1

    # Update theta_best
    if J_test < J_min:
        theta_best = theta_k.copy()
        J_min = J_test


def main(max_iter):
    """ Artificial Neural Network for classifying galaxies with multiprocessing optimization """
    start_time = time()
    logger.info("Starting neural network training with multiprocessing optimization...")
    logger.info(f"max iter={max_iter}")

    # Set random seed
    np.random.seed(917)

    # Load data using pandas for faster loading
    train = pd.read_csv('train.csv').values
    test = pd.read_csv('test.csv').values

    logger.info(f"Data loaded in {time() - start_time:.2f}s")

    # Get labels and normalize data
    train_label = train[:, 0].reshape(len(train), 1)
    test_label = test[:, 0].reshape(len(test), 1)
    train = train[:, 1:] / 255.
    test = test[:, 1:] / 255.
    X = train
    y = train_label

    # Network parameters
    m = np.shape(X)[0]
    input_layer_size = np.shape(X)[1]
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0

    logger.info(f"Network architecture: {input_layer_size}->{hidden_layer_size}->{num_labels}")
    logger.info(f"Using {mp.cpu_count()} CPU cores for parallel processing")

    # Initialize weights
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 0.4 - 0.2
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 0.4 - 0.2
    theta0 = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    # Initial evaluation
    J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    logger.info(f'Initial cost function J = {J}')

    train_pred = predict(Theta1, Theta2, train)
    initial_accuracy = np.sum(1. * (train_pred == train_label)) / len(train_label)
    logger.info(f'Initial accuracy on training set = {initial_accuracy}')

    # Initialize tracking arrays
    global Js_train, Js_test, N_iter, J_min, theta_best
    Js_train = np.array([J])
    J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    Js_test = np.array([J_test])
    N_iter = 1
    J_min = np.inf
    theta_best = theta0.copy()

    # Optimization
    logger.info("Starting optimization...")
    opt_start_time = time()

    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf = partial(callbackF, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label)
    theta = optimize.fmin_cg(cost_function, theta0, fprime=gradient, args=args, callback=cbf, maxiter=max_iter)

    logger.info(f"Optimization completed in {time() - opt_start_time:.2f}s")

    # Final evaluation
    Theta1, Theta2 = reshape(theta_best, input_layer_size, hidden_layer_size, num_labels)
    train_pred = predict(Theta1, Theta2, train)
    test_pred = predict(Theta1, Theta2, test)

    final_train_accuracy = np.sum(1. * (train_pred == train_label)) / len(train_label)
    final_test_accuracy = np.sum(1. * (test_pred == test_label)) / len(test_label)
    logger.info(f'Final accuracy on training set = {final_train_accuracy:.4f}')
    logger.info(f'Final accuracy on test set = {final_test_accuracy:.4f}')

    total_time = time() - start_time
    logger.info(f"Total training time: {total_time:.2f}s")

    return final_test_accuracy, total_time


if __name__ == "__main__":
    main(max_iter=50)