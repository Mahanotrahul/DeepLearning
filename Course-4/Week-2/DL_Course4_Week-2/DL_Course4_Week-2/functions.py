

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
        Y = np.vstack([Y, new_rows]).T

        for row in range(Y.shape[0]):
            value = int(Y[row, 0])
            Y[row, 0] = 0
            Y[row, value] = 1
        
        dictt[str(i)] = Y
        del Y
    return dictt

#def random_mini_batches(X, Y, mini_batch_size = 256, seed = 10):
#    m = X.shape[1]


#    # Step 1: Shuffle (X, Y)
#    permutation = list(np.random.permutation(m))
#    X_shuffled = X[:, permutation]
#    Y_shuffled = Y[:, permutation].reshape((Y.shape[0],m))

#    mini_batches = []
#    np.random.seed(seed)

#    complete_mini_batches = math.floor(m / mini_batch_size)
#    for i in range(0 ,complete_mini_batches):
#        mini_batch_X = X_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]
#        mini_batch_Y = Y_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]

#        mini_batch = (mini_batch_X, mini_batch_Y)
#        mini_batches.append(mini_batch)

#    i = complete_mini_batches
#    if m%mini_batch_size != 0:
#        mini_batch_X = X_shuffled[:, i*mini_batch_size : ]
#        mini_batch_Y = Y_shuffled[:, i*mini_batch_size : ]

#        mini_batch = (mini_batch_X, mini_batch_Y)
#        mini_batches.append(mini_batch)

#    return mini_batches

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
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



