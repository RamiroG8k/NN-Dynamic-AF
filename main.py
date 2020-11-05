import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
# Dataset
from sklearn.datasets import make_circles

# Neural layers
from neural_layer import Neural_Layer

'''
Anonymous Sigmoid activation function and derivate
index 0 actual function
index 1 derivation function
'''
# sigm = (lambda x: 1 / (1 + np.e ** (-x)),
#         lambda x: x * (1 - x))
def sigm(x, derivate=False):
    u =  1 / (1 + np.e ** (-x))
    du = x * (1 - x)
    return du if (derivate==True) else u
    

def create_nn(topology, act_f):
    nn = []
    for i, connections in enumerate(topology[:-1]):
        nn.append(Neural_Layer(connections, topology[i+1], act_f))
    return nn

if __name__ == "__main__":
    '''
    Create dataset
    n   stands for number of samples to train with
    p   number of inputs per sample, characteristics of every sample
    '''
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
    print(create_nn(topology, sigm))

    # plt.axis("equal")
    # plt.show()
