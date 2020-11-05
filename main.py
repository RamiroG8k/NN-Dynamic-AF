'''
Algorithm: Neural Network with dynamic activation functions
Date: Thursday November 5th, 2020
Autor: Ramiro Mendez, based in Carlos Santana Model
'''
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles   # Dataset

# Neural layers
from neural_layer import Neural_Layer

'''
Anonymous function (commented) 
    Sigmoid activation function and derivate
    index 0 actual function
    index 1 derivation function
'''
# sigm = (lambda x: 1 / (1 + np.e ** (-x)),
#         lambda x: x * (1 - x))


def sigm(x, derivate=False):
    u = 1 / (1 + np.e ** (-x))
    du = x * (1 - x)
    return du if (derivate == True) else u


'''
Cost Function (anonymous commented) 
    Least Mean Square function and derivate
    Returns difference between output and desired 
'''
# lms_cost = (lambda output, desired: np.mean((output - desired) ** 2),
#             lambda output, desired: (output - desired))


def lms_cost(output, desired, derivate=False):
    u = np.mean((output - desired) ** 2)
    du = np.mean(output - desired)
    return du if (derivate == True) else u


'''
Neural Network topology
    Returns nn  Neural network array with layers inside
    topology    Vector to instantiate n neurons per layer in NN
    act_f       Activation function

    For item in array 'topology' creates actual layer 
    Instantiates 'n' inputs to the next one and 'act_f' per layer
'''


def create_nn(topology, act_f):
    nn = []
    for i, connections in enumerate(topology[:-1]):
        nn.append(Neural_Layer(connections, topology[i+1], act_f))
    return nn


def train(neural_network, samples, desired, lms_cost, lr=0.5):
    '''
    Forward pass -> 
        Gets samples and desired, passes through layers
        Applies weighted sum (samples * weights + bias)
        Weighted sum to Activation function
    '''
    output = [(None, samples)]
    for _, layer in enumerate(neural_network):
        # @ it is used to matrix multiplication
        z = output[-1][1] @ layer.weights + layer.bias
        a = layer.activation_f(z)
        output.append((z, a))

    print(output[-1][1])


if __name__ == "__main__":
    # Create dataset
    # n   stands for number of samples to train with
    # p   number of inputs per sample, characteristics of every sample
    n = 500
    p = 2

    '''
    make_circles
        n_samples : int or two-element tuple, optional (default=100)
            If int, it is the total number of points generated. 
            For odd numbers, the inner circle will have one point more than the outer circle. 
            If two-element tuple, number of points in outer circle and inner circle.

        noise : double or None (default=None)
            Standard deviation of Gaussian noise added to the data.

        factor : 0 < double < 1 (default=.8)
            Scale factor between inner and outer circle.    
    '''
    # Samples it's a matrix that gets inputs per sample
    # Desired it's a vector that stores clasification per sample
    samples, desired = make_circles(n_samples=n, factor=0.5, noise=0.065)

    # Samples in first class, in this case class  '0'
    plt.scatter(samples[desired == 0, 0],
                samples[desired == 0, 1], color="green")

    # Samples in first class, in this case class  '1'
    plt.scatter(samples[desired == 1, 0],
                samples[desired == 1, 1], color="red")
    '''
    Stores number of neurons per layer in neural network
    Final is 1 because classification is binary, based in 0 or 1
    '''
    topology = [p, 4, 8, 16, 8, 4, 1]
    neural_network = create_nn(topology, sigm)

    train(neural_network, samples, desired, lms_cost)

    # plt.axis("equal")
    # plt.show()
