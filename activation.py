import numpy as np

'''
Anonymous function (commented) 
    Sigmoid activation function and derivate
'''
# sigm = (lambda x: 1 / (1 + np.e ** (-x)),
#         lambda x: x * (1 - x))
def sigm(x, derivate=False):
    u = 1 / (1 + np.e ** (-x))
    du = x * (1 - x)
    return du if (derivate == True) else u


'''
Anonymous function (commented) 
    Relu activation function and derivate
'''
# relu = lambda x: np.maximum(0,x)

'''
Cost Function (anonymous commented) 
    Least Mean Square function and derivate
    Returns difference between output and desired 
'''
# lms_cost = (lambda output, desired: np.mean((output - desired) ** 2),
#             lambda output, desired: (output - desired))


def lms_cost(output, desired, derivate=False):
    u = np.mean((output - desired) ** 2)
    du = (output - desired)
    return du if (derivate == True) else u


'''
TODO: activation functions for output layer
'''
def linear(z, derivative = False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a
def logistic(z, derivative = False):
    a =  (1)/ (1 + np.exp(-z))
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def softmax(z, derivative = False):
    e = np.exp(z - np.max(z, axis = 0))
    a = (e) / (np.sum(e, axis = 0))
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

'''
TODO: activation functions for hidden layers
'''
def tanh(z, derivative = False):
    a = np.tanh(z)
    if derivative:
        da = (1 - a) * (1 + a)
        return a, da
    return a

def relu(z, derivative = False):
    a = (z) * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype= float)
        return a, da
    return a

def logistic_hidden(z, derivative = False):
    a = (1)/(1 + np.exp(-z))
    if derivative:
        da = (a) * (1 - a)
        return a, da
    return a