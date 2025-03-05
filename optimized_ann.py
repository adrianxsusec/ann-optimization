import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
from line_profiler import profile
import multiprocessing as mp
import torch
from numba import jit
import logging
from time import time
import pandas as pd

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


def predict(Theta1, Theta2, X, use_torch=False):
    """ Predict labels in a trained three layer classification network.
    Input:
      Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
      Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
      X            matrix of training data      (m x input_layer_size)
    Output:
      prediction   label prediction
    """
    if use_torch:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device('cuda')
            print("device: ", device)
        else:
            device = torch.device('cpu')

        # torch version
        X_gpu = torch.from_numpy(X).to(device)
        Theta1_gpu = torch.from_numpy(Theta1).to(device)
        Theta2_gpu = torch.from_numpy(Theta2).to(device)

        m = X_gpu.shape[0]
        a1 = torch.cat([torch.ones(m, 1).to(device), X_gpu], dim=1)
        a2 = torch.sigmoid(a1 @ Theta1_gpu.T)
        a2 = torch.cat([torch.ones(m, 1).to(device), a2], dim=1)
        a3 = torch.sigmoid(a2 @ Theta2_gpu.T)

        prediction = torch.argmax(a3, dim=1).cpu().numpy().reshape((m, 1))

        # Clean up GPU memory
        if cuda_available:
            torch.cuda.empty_cache()
    else:
        print("no cuda used")
        # Original CPU version
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


def cost_function(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, use_torch=False):
    """ Neural net cost function with GPU acceleration option """
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
    m = len(y)

    if use_torch:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device('cuda')
            print("device: ", device)
        else:
            device = torch.device('cpu')

        # torch version
        X_gpu = torch.from_numpy(X).to(device)
        Theta1_gpu = torch.from_numpy(Theta1).to(device)
        Theta2_gpu = torch.from_numpy(Theta2).to(device)

        # Forward pass
        a1 = torch.cat([torch.ones(m, 1).to(device), X_gpu], dim=1)
        a2 = torch.sigmoid(a1 @ Theta1_gpu.T)
        a2 = torch.cat([torch.ones(m, 1).to(device), a2], dim=1)
        a3 = torch.sigmoid(a2 @ Theta2_gpu.T)

        # One-hot encode y
        y_mtx = torch.zeros(m, num_labels).to(device)
        y_mtx[torch.arange(m), y.flatten()] = 1

        # Cost computation
        J = (-y_mtx * torch.log(a3) - (1 - y_mtx) * torch.log(1 - a3)).sum() / m

        # Add regularization
        J += lmbda / (2 * m) * (torch.sum(Theta1_gpu[:, 1:] ** 2) + torch.sum(Theta2_gpu[:, 1:] ** 2))

        result = J.cpu().item()

        # Clean up GPU memory
        if cuda_available:
            torch.cuda.empty_cache()
        return result
    else:
        # Original CPU version
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


def gradient_worker(batch_indices, X, y, Theta1, Theta2, num_labels, use_torch=False):
    """Worker function for parallel gradient computation"""


    if use_torch:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        X_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(y).to(device)
        Theta1_tensor = torch.from_numpy(Theta1).to(device)
        Theta2_tensor = torch.from_numpy(Theta2).to(device)

        t = len(batch_indices)

        # Forward pass
        # directly using sigmoid here instead of g()
        a1 = torch.cat((torch.ones(t, 1).to(device), X_tensor[batch_indices, :]), dim=1)
        z2 = torch.matmul(a1, Theta1_tensor.T)
        a2 = torch.sigmoid(z2)
        a2 = torch.cat((torch.ones(t, 1).to(device), a2), dim=1)
        z3 = torch.matmul(a2, Theta2_tensor.T)
        a3 = torch.sigmoid(z3)

        # Backward pass
        y_mtx = torch.zeros(t, num_labels, dtype=torch.float32).to(device)
        y_mtx[torch.arange(t), y_tensor[batch_indices].flatten().long()] = 1
        delta3 = a3 - y_mtx
        delta2 = torch.matmul(delta3, Theta2_tensor[:, 1:]) * (a2[:, 1:] * (1 - a2[:, 1:]))
        Delta2 = torch.matmul(delta3.T, a2).cpu().numpy()
        Delta1 = torch.matmul(delta2.T, a1).cpu().numpy()

    else:
        Delta1 = np.zeros_like(Theta1)
        Delta2 = np.zeros_like(Theta2)

        for t in batch_indices:
            # Forward pass
            a1 = X[t, :].reshape((Theta1.shape[1] - 1, 1))
            a1 = np.vstack((1, a1))
            z2 = Theta1 @ a1
            a2 = g(z2)
            a2 = np.vstack((1, a2))
            a3 = g(Theta2 @ a2)

            # Backward pass
            y_k = np.zeros((num_labels, 1))
            y_k[y[t, 0].astype(int)] = 1
            delta3 = a3 - y_k
            delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)

            Delta2 += delta3 @ a2.T
            Delta1 += delta2 @ a1.T

    return Delta1, Delta2

@profile
def native_gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Neural net cost function gradient for a three layer classification network.
    Input:
      theta               flattened vector of neural net model parameters
      input_layer_size    size of input layer
      hidden_layer_size   size of hidden layer
      num_labels          number of labels
      X                   matrix of training data
      y                   vector of training labels
      lmbda               regularization term
    Output:
      grad                flattened vector of derivatives of the neural network
    """

    # unflatten theta
    Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)

    # number of training values
    m = len(y)

    # Backpropagation: calculate the gradients Theta1_grad and Theta2_grad:

    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((num_labels, hidden_layer_size + 1))

    for t in range(m):
        # forward
        a1 = X[t, :].reshape((input_layer_size, 1))
        a1 = np.vstack((1, a1))  # +bias
        z2 = Theta1 @ a1
        a2 = g(z2)
        a2 = np.vstack((1, a2))  # +bias
        a3 = g(Theta2 @ a2)

        # compute error for layer 3
        y_k = np.zeros((num_labels, 1))
        y_k[y[t, 0].astype(int)] = 1
        delta3 = a3 - y_k
        Delta2 += (delta3 @ a2.T)

        # compute error for layer 2
        delta2 = (Theta2[:, 1:].T @ delta3) * grad_g(z2)
        Delta1 += (delta2 @ a1.T)

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    # add regularization
    Theta1_grad[:, 1:] += (lmbda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lmbda / m) * Theta2[:, 1:]

    # flatten gradients
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return grad

@profile
def distributed_gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    """ Parallel gradient computation using multiprocessing """
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
            worker = partial(gradient_worker,
                             X=X, y=y,
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
    if (J_test < J_min):
        theta_best = theta_k
        J_min = J_test

    # Update Plot
    if plt.get_fignums():  # Only if figure exists
        iters = np.arange(len(Js_train))
        # plt.clf()
        # plt.subplot(2, 1, 1)
        im_size = 32
        pad = 4
        galaxies_image = np.zeros((3 * im_size, 6 * im_size + 2 * pad), dtype=int) + 255
        for i in range(3):
            for j in range(6):
                idx = 3 * j + i + 900 * (j > 1) + 900 * (j > 3) + (N_iter % 600)
                shift = 0 + pad * (j > 1) + pad * (j > 3)
                ii = i * im_size
                jj = j * im_size + shift
                galaxies_image[ii:ii + im_size, jj:jj + im_size] = X[idx].reshape(im_size, im_size) * 255
                my_label = 'E' if y_pred[idx] == 0 else 'S' if y_pred[idx] == 1 else 'I'
                my_color = 'blue' if (y_pred[idx] == y[idx]) else 'red'
                # plt.text(jj + 2, ii + 10, my_label, color=my_color)
                if (y_pred[idx] == y[idx]):
                    plt.text(jj + 4, ii + 25, "✓", color='blue', fontsize=50)
        # plt.imshow(galaxies_image, cmap='gray')
        # plt.gca().axis('off')
        # plt.subplot(2, 1, 2)
        # plt.plot(iters, Js_test, 'r', label='test')
        # plt.plot(iters, Js_train, 'b', label='train')
        # plt.xlabel("iteration")
        # plt.ylabel("cost")
        # plt.xlim(0, 600)
        # plt.ylim(1, 2.1)
        # plt.gca().legend()
        # plt.pause(0.001)


@profile
def main():
    """ Artificial Neural Network for classifying galaxies with GPU and parallel processing """
    start_time = time()
    logger.info("Starting neural network training...")

    # Set random seed
    np.random.seed(917)

    train = pd.read_csv('../ann-optimization/train.csv').values
    test = pd.read_csv('../ann-optimization/test.csv').values

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
    global Js_train, Js_test
    Js_train = np.array([J])
    J_test = cost_function(theta0, input_layer_size, hidden_layer_size, num_labels, test, test_label, lmbda)
    Js_test = np.array([J_test])

    # Prep visualization
    if not plt.get_fignums():
        plt.figure(figsize=(6, 6), dpi=80)

    # Optimization
    logger.info("Starting optimization...")
    opt_start_time = time()

    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    cbf = partial(callbackF, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda, test, test_label)
    theta = optimize.fmin_cg(cost_function, theta0, fprime=distributed_gradient, args=args, callback=cbf, maxiter=50)

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

    # Save figure if it exists
    if plt.get_fignums():
        plt.savefig('artificialneuralnetwork.png', dpi=240)
        # # plt.show()


if __name__ == "__main__":
    main()