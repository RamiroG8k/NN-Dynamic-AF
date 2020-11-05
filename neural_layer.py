import numpy as np

# Class for layers in neural network
class Neural_Layer(object):
    '''
    Structure of data
    n_connections   As many as neurons in previous layer
    n_neurons       Neurons in actual layer
    activation_f    Activation function for neurons in layer
    bias            As many as neurons in actual layer
    '''
    def __init__(self, n_connections, n_neurons, activation_f):
        self.activation_f = activation_f
        # * 2 - 1  is used to get values in range (-1, 1)
        self.bias = np.random.rand(1, n_neurons) * 2 - 1
        self.weights = np.random.rand(n_connections, n_neurons) * 2 - 1 
