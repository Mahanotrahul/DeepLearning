import numpy as np

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