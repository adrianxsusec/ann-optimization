import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
import torch
from numba import jit
import logging
from time import time

matplotlib.use("TkAgg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-allocated GPU tensors
X_train_gpu = None
y_train_gpu = None
X_test_gpu = None
y_test_gpu = None
X_train_shape = None
X_test_shape = None


def initialize_gpu_tensors(X, y):
    """Pre-allocate tensors on GPU once"""
    X_gpu = torch.from_numpy(X).to(device)
    y_gpu = torch.from_numpy(y).to(device)
    return X_gpu, y_gpu


def g(x):
    """Sigmoid function (not needed for GPU path, kept for CPU fallback)"""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    else:
        return 1.0 / (1.0 + np.exp(-x))


def grad_g(x):
    """Gradient of sigmoid function (not needed for GPU path, kept for CPU fallback)"""
    if isinstance(x, torch.Tensor):
        gx = torch.sigmoid(x)
        return gx * (1.0 - gx)
    else:
        gx = g(x)
        return gx * (1.0 - gx)


def predict(Theta1, Theta2, X):
    """Predict labels using GPU acceleration"""
    global X_train_gpu, X_test_gpu, X_train_shape, X_test_shape

    m = X.shape[0]

    if torch.cuda.is_available():
        # Choose appropriate pre-allocated tensor based on shape
        if X_train_shape is not None and np.array_equal(X.shape, X_train_shape):
            X_gpu = X_train_gpu
        elif X_test_shape is not None and np.array_equal(X.shape, X_test_shape):
            X_gpu = X_test_gpu
        else:
            X_gpu = torch.from_numpy(X).to(device)

        # Convert weights to GPU tensors
        Theta1_gpu = torch.from_numpy(Theta1).to(device)
        Theta2_gpu = torch.from_numpy(Theta2).to(device)

        with torch.no_grad():  # Disable gradient calculation for prediction
            # Forward pass
            a1 = torch.cat([torch.ones(m, 1, device=device), X_gpu], dim=1)
            a2 = torch.sigmoid(torch.matmul(a1, Theta1_gpu.T))
            a2 = torch.cat([torch.ones(m, 1, device=device), a2], dim=1)
            a3 = torch.sigmoid(torch.matmul(a2, Theta2_gpu.T))

            # Get prediction and convert back to numpy
            prediction = torch.argmax(a3, dim=1).cpu().numpy().reshape((m, 1))

        return prediction
    else:
        # CPU fallback
        a1 = np.hstack((np.ones((m, 1)), X))
        a2 = g(a1 @ Theta1.T)
        a2 = np.hstack((np.ones((m, 1)), a2))
        a3 = g(a2 @ Theta2.T)
        prediction = np.argmax(a3, 1).reshape((m, 1))
        return prediction


def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
    """Reshape theta into Theta1 and Theta2"""
    ncut = hidden_layer_size * (input_layer_size + 1)
    Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size + 1)
    return Theta1, Theta2


def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """Neural net cost function with GPU acceleration"""
    global X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu, X_train_shape, X_test_shape

    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    if torch.cuda.is_available():
        # Choose appropriate pre-allocated tensor based on shape
        if X_train_shape is not None and np.array_equal(X.shape, X_train_shape):
            X_gpu = X_train_gpu
            y_gpu = y_train_gpu
        elif X_test_shape is not None and np.array_equal(X.shape, X_test_shape):
            X_gpu = X_test_gpu
            y_gpu = y_test_gpu
        else:
            X_gpu = torch.from_numpy(X).to(device)
            y_gpu = torch.from_numpy(y).to(device)

        # Convert weights to GPU tensors
        Theta1_gpu = torch.from_numpy(Theta1).to(device)
        Theta2_gpu = torch.from_numpy(Theta2).to(device)

        # Forward pass
        a1 = torch.cat([torch.ones(m, 1, device=device), X_gpu], dim=1)
        a2 = torch.sigmoid(torch.matmul(a1, Theta1_gpu.T))
        a2 = torch.cat([torch.ones(m, 1, device=device), a2], dim=1)
        a3 = torch.sigmoid(torch.matmul(a2, Theta2_gpu.T))

        # One-hot encode y
        y_mtx = torch.zeros(m, num_labels, device=device)
        y_mtx[torch.arange(m), y_gpu.flatten().long()] = 1

        # Compute cost
        J = torch.sum(-y_mtx * torch.log(a3) - (1 - y_mtx) * torch.log(1 - a3)) / m

        # Add regularization
        reg_term = (torch.sum(Theta1_gpu[:, 1:] ** 2) + torch.sum(Theta2_gpu[:, 1:] ** 2)) * (lmbda / (2 * m))
        J += reg_term

        return J.item()  # Convert to Python scalar
    else:
        # CPU fallback
        a1 = np.hstack((np.ones((m, 1)), X))
        a2 = g(a1 @ Theta1.T)
        a2 = np.hstack((np.ones((m, 1)), a2))
        a3 = g(a2 @ Theta2.T)

        # One-hot encode y
        y_mtx = np.zeros((m, num_labels))
        y_mtx[np.arange(m), y.flatten().astype(int)] = 1

        J = np.sum(-y_mtx * np.log(a3) - (1 - y_mtx) * np.log(1 - a3)) / m
        J += lmbda / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

        return J


def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """Gradient computation with GPU acceleration"""
    global X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu, X_train_shape, X_test_shape

    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    if torch.cuda.is_available():
        # Choose appropriate pre-allocated tensor based on shape
        if X_train_shape is not None and np.array_equal(X.shape, X_train_shape):
            X_gpu = X_train_gpu
            y_gpu = y_train_gpu
        elif X_test_shape is not None and np.array_equal(X.shape, X_test_shape):
            X_gpu = X_test_gpu
            y_gpu = y_test_gpu
        else:
            X_gpu = torch.from_numpy(X).to(device)
            y_gpu = torch.from_numpy(y).to(device)

        # Convert weights to GPU tensors
        Theta1_gpu = torch.from_numpy(Theta1).to(device)
        Theta2_gpu = torch.from_numpy(Theta2).to(device)

        # Vectorized forward pass for all samples
        a1 = torch.cat([torch.ones(m, 1, device=device), X_gpu], dim=1)
        z2 = torch.matmul(a1, Theta1_gpu.T)
        a2 = torch.sigmoid(z2)
        a2 = torch.cat([torch.ones(m, 1, device=device), a2], dim=1)
        a3 = torch.sigmoid(torch.matmul(a2, Theta2_gpu.T))

        # One-hot encode y
        y_one_hot = torch.zeros(m, num_labels, device=device)
        y_one_hot.scatter_(1, y_gpu.long(), 1)

        # Backpropagation - vectorized across all samples
        delta3 = a3 - y_one_hot
        delta2 = torch.matmul(delta3, Theta2_gpu[:, 1:]) * (a2[:, 1:] * (1.0 - a2[:, 1:]))

        # Compute gradients
        Delta1 = torch.matmul(delta2.T, a1)
        Delta2 = torch.matmul(delta3.T, a2)

        # Apply regularization
        Theta1_grad = Delta1 / m
        Theta2_grad = Delta2 / m
        Theta1_grad[:, 1:] += (lmbda / m) * Theta1_gpu[:, 1:]
        Theta2_grad[:, 1:] += (lmbda / m) * Theta2_gpu[:, 1:]

        # Convert back to numpy and flatten
        grad = np.concatenate((Theta1_grad.cpu().numpy().flatten(),
                               Theta2_grad.cpu().numpy().flatten()))

        return grad
    else:
        # CPU fallback
        Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
        Delta2 = np.zeros((num_labels, hidden_layer_size + 1))

        # Loop through samples
        for t in range(m):
            # Forward pass
            a1 = X[t, :].reshape((input_layer_size, 1))
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


def main():
    """ Artificial Neural Network for classifying galaxies with GPU optimization """
    global X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu, X_train_shape, X_test_shape
    global N_iter, J_min, theta_best, Js_train, Js_test

    start_time = time()
    logger.info("Starting neural network training with GPU optimization...")

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

    # Save shapes for later tensor matching
    X_train_shape = X.shape
    X_test_shape = test.shape

    # Network parameters
    m = np.shape(X)[0]
    input_layer_size = np.shape(X)[1]
    hidden_layer_size = 8
    num_labels = 3
    lmbda = 1.0

    logger.info(f"Network architecture: {input_layer_size}->{hidden_layer_size}->{num_labels}")

    # Initialize weights
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 0.4 - 0.2
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 0.4 - 0.2
    theta0 = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    # Pre-allocate GPU tensors
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        X_train_gpu, y_train_gpu = initialize_gpu_tensors(X, y)
        X_test_gpu, y_test_gpu = initialize_gpu_tensors(test, test_label)
    else:
        logger.info("No GPU available, using CPU")

    # Initial evaluation
    J = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    logger.info(f'Initial cost function J = {J}')

    train_pred = predict(Theta1, Theta2, train)
    initial_accuracy = np.sum(1. * (train_pred == train_label)) / len(train_label)
    logger.info(f'Initial accuracy on training set = {initial_accuracy}')

    # Initialize tracking arrays
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
    theta = optimize.fmin_cg(cost_function, theta0, fprime=gradient, args=args, callback=cbf, maxiter=50)

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

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test_accuracy, total_time


if __name__ == "__main__":
    main()