

import numpy as np
import math

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    """

    A = 1/(1 + np.exp(-Z))
    return A


def relu(Z):

    s = np.maximum(0,Z)
    
    return s


def sigmoid_backward(dA, Z):
   

    s = sigmoid(Z)
    dZ = dA*s*(1 - s)


    return dZ

def relu_backward(dA, Z):

    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    dZ = np.array(dA, copy = True)          ## just converting dz to a correct object.

    # When Z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0


    return dZ

def tanh_backward(dA, Z):
    s = np.tanh(Z)

    dZ = dA*(1 - np.power(s, 2))

    return dZ

def one_hot_encoding(dict):
    #print(np.amax(Y))
    dictt = {}
    for i in dict:
        Y = dict[i]
        new_rows = np.zeros((np.amax(Y), Y.shape[1]))
        Y = np.vstack([Y, new_rows])

        for col in range(Y.shape[1]):
            value = int(Y[0,col])
            Y[0,col] = 0
            Y[value,col] = 1
        
        dictt[str(i)] = Y
        del Y
    return dictt

def random_mini_batches(X, Y, mini_batch_size = 256, seed = 10):
    m = X.shape[1]


    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation].reshape((Y.shape[0],m))

    mini_batches = []
    np.random.seed(seed)

    complete_mini_batches = math.floor(m / mini_batch_size)
    for i in range(0 ,complete_mini_batches):
        mini_batch_X = X_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]
        mini_batch_Y = Y_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    i = complete_mini_batches
    if m%mini_batch_size != 0:
        mini_batch_X = X_shuffled[:, i*mini_batch_size : ]
        mini_batch_Y = Y_shuffled[:, i*mini_batch_size : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def normalize(X , X_test):
    # Normalize Inputs
    # Normailze Mean
    X -= np.mean(X , axis = 0)
    X_test -= np.mean(X_test, axis = 0)

    # Normalize Variance
    X /= np.var(X, axis = 0)**2
    X_test /= np.var(X_test, axis = 0)**2

    X -= np.mean(X)
    X_test -= np.mean(X)
        
    X /= np.var(X)**2
    X_test /= np.var(X)**2

    return X, X_test