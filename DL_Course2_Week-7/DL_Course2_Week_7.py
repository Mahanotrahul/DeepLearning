

import time
start_time = time.time()

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import scipy.io as sio
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets, load_dataset, load_dataset_SIGNS
from functions import sigmoid, relu, sigmoid_backward, relu_backward, tanh_backward, one_hot_encoding, random_mini_batches, normalize

#import mnist

#images = mnist.train_images()
#x = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
#print(x)
#print(x.shape)
#plt.imshow(x[1][:])
#plt.show()


try:
    override = input("override(0 for no and any other option for yes)?\t")
    if override == "0" or override == 0:
        override = False
    else:
        override = True
except:
    print("Wrong input")
    override = 1

if override == True:
    print("Over Riding all input requirements with default values")

# Datasets
if override == 0:

    dataset_option = input("Which Dataset you want to run the NN Model?X: Image-Classification\nS : SIGNS Dataset\nN: Hand-Written Digit Classification\nNb : Hand-Written Digits BigDatset [MNIST Datset]\t")
    if dataset_option == "X":
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
        train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
        print(train_set_x.shape)
        print(test_set_x_orig.shape)
        test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
        print(test_set_x.shape)
        num_px = train_set_x_orig.shape[1]

        X = train_set_x/255
        Y = train_set_y
        X_test = test_set_x/255
        Y_test = test_set_y

        print(Y_test)
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))
        dict = one_hot_encoding(dict = {"Y" : Y, "Y_test" : Y_test})
        Y = dict["Y"]
        Y_test = dict["Y_test"]
        print(Y_test)


        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))


    elif dataset_option == "S":
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset_SIGNS()
        train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
        print(train_set_x.shape)
        print(test_set_x_orig.shape)
        test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
        print(test_set_x.shape)
        num_px = train_set_x_orig.shape[1]
        
        X = train_set_x/255
        Y = train_set_y
        X_test = test_set_x/255
        Y_test = test_set_y
        print(Y_test)

        # One Hot Encoding
        dict = {'Y' : Y, 
                'Y_test' : Y_test}
        dict = one_hot_encoding(dict)
        Y = dict['Y']
        Y_test= dict["Y_test"]
        del dict
        print(Y)

        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))


    elif dataset_option == "N":

        #test = sio.loadmat('datasets/Digit_Classification-BigDataset.mat')
        #X = test['X'][:]
        #Y = test['Y'][:]
        #X_test = test['X_test'][:]
        #Y_test = test['Y_test'][:]

        #from tensorflow.examples.tutorials.mnist import input_data
        #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #X_train = (np.vstack([img.reshape(-1,) for img in mnist.train.images])).T
        #Y_train = (mnist.train.labels).T
        #X_test = (np.vstack([img.reshape(-1,) for img in mnist.test.images])).T
        #Y_test = (mnist.test.labels).T
        #X = X_train
        #Y = Y_train
        #print("Y_test.shape" + str(Y_test.shape))
        #print("X_test" + str(X_test.shape))
        #print("X_train" + str(X_train.shape))
        #print("Y_train" + str(Y_train.shape))

        #sio.savemat("datasets/Digit_Classification-BigDataset.mat", {"X": X, "Y": Y, "X_test" : X_test, "Y_test" : Y_test})
        #del mnist


        test = sio.loadmat('..\\..\\..\\..\\..\\datasets\\Digit_Classification.mat')
        X = test['X'][:]
        Y = test['Y'][:]

        #X, Y = X.T, Y.T
        sel = np.arange(X.shape[1])
        np.random.shuffle(sel)
        print(sel)
        print(sel.shape)
        set_divide = (90*X.shape[1])//100
        X_train = X[:,sel[0:set_divide]]
        X_test = X[:,sel[set_divide:X.shape[1]]]
        X = X_train




        ##for i in range(Y.shape[1]):
        ##    if Y[:,i] == 10:
        ##        Y[:,i] = 0
        ##print(np.amax(Y))
        ##new_rows = np.zeros((np.amax(Y), Y.shape[1]))
        ##Y = np.vstack([Y, new_rows])
        ##print(Y.shape)

        ##for i in range(Y.shape[1]):
        ##    c = i//500
        ##    Y[c,i] = 1
        ##    Y[0,i] = 0
        ##    Y[np.argmax(Y[:,i]), i] = 1
        ##sio.savemat("datasets/Digit_Classification.mat", {"X": X, "Y": Y})
        Y_train = Y[:,sel[0:set_divide]]
        Y_test = Y[:,sel[set_divide:Y.shape[1]]]
        Y = Y_train
        print(Y.shape)
        print(Y_test.shape)

        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))
    
    elif dataset_option == "Nb":
        test = sio.loadmat('..\\..\\..\\..\\..\\datasets\\Digit_Classification-BigDataset.mat')
        X = test['X'][:]
        Y = test['Y'][:]
        X_test = test['X_test'][:]
        Y_test = test['Y_test'][:]


        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))

    else:
        print("Wrong Argument. Using Default dataset gaussian_quantiles")
        import sys
        sys.exit()

    
else:
    dataset_option = "S"
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset_SIGNS()
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    print(train_set_x.shape)
    print(test_set_x_orig.shape)
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    print(test_set_x.shape)
    num_px = train_set_x_orig.shape[1]
        
    X = train_set_x/255
    Y = train_set_y
    X_test = test_set_x/255
    Y_test = test_set_y
    print(Y_test)

    # One Hot Encoding
    dict = {'Y' : Y, 
            'Y_test' : Y_test}
    dict = one_hot_encoding(dict)
    Y = dict['Y']
    Y_test= dict["Y_test"]
    del dict
    print(Y)

    print("Y.shape : " + str(Y.shape))
    print("X.shape : " + str(X.shape))
    print("Y_test.shape : " + str(Y_test.shape))
    print("X_test.shape : " + str(X_test.shape))
  
    



#Example of a picture
if override == 0:
    if dataset_option == "X":
        try:
            index = int(input("Index of the picture( 0 -  %i )  ? " %X.shape[1]))
            print("Selected Index Number : %i" %index)
            print(str(train_set_y[:,index]) + "It's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture. ")
            plt.imshow(train_set_x_orig[index])
            plt.show()

        except Exception as e:
            print(e)
    
    elif dataset_option == "S":
        try:
            index = int(input("Index of the picture( 0 -  %i )  ? " %X.shape[1]))
            print("Selected Index Number : %i" %index)
            plt.imshow(train_set_x_orig[index])
            plt.title("Number: " + str(np.argmax(Y[:,index])))
            plt.xlabel(Y[:,index])
            plt.show()
        except Exception as e:
            print(e)
        
    elif dataset_option == "N":
        sel = np.random.randint(1, X.shape[1])
        plt.imshow(X[:,sel].reshape(20, 20), cmap = 'gray_r')
        plt.title("Number: " + str(np.argmax(Y[:,sel])))
        plt.xlabel(Y[:,sel])
        plt.show()
    
    elif dataset_option == "Nb":
        sel = np.random.randint(1, X.shape[1])
        plt.imshow(X[:,sel].reshape(28, 28), cmap = 'gray_r')
        plt.title("Number: " + str(np.argmax(Y[:,sel])))
        plt.xlabel(Y[:,sel])
        plt.show()

    else:
        # Visualize the data
        plt.scatter(X[0, :], X[1, :],  s=40, c = Y[0,:] , cmap=plt.cm.Spectral);
        plt.show()

        # Train the logistic regression classifier
        clf = sklearn.linear_model.LogisticRegressionCV();
        clf.fit(X.T, Y.T);

        # Plot the decision boundary for logistic regression
        plot_decision_boundary(lambda x: clf.predict(x), X, Y)
        plt.title("Logistic Regression")
        plt.show()

        # Print accuracy
        LR_predictions = clf.predict(X.T)
        print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
               '% ' + "(percentage of correctly labelled datapoints)")


def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype = tf.float32, shape = (n_x, None))
    Y = tf.placeholder(dtype = tf.float32, shape = (n_y, None))

    return X, Y

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x, n_y

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1

    tf.set_random_seed(1)
    for l in range(1, L + 1):
        parameters["W" + str(l)] = tf.get_variable(name = "W" + str(l), shape = (layer_dims[l], layer_dims[l - 1]), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b" + str(l)] = tf.get_variable(name = "b" + str(l), shape = (layer_dims[l], 1), dtype = tf.float32, initializer = tf.zeros_initializer())
    return parameters

def initialize_momentum(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(1 , L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
    return v

def initialize_rmsprop(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(1 , L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))

    return v

def initialize_adams(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(1 , L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
        s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0], parameters["W" + str(l)].shape[1]))
        s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0], parameters["b" + str(l)].shape[1]))
    return v, s


def forward_prop(X, parameters, activation_func):
    cache = {}
    activations = {}
    activations["A" + str(0)] = X
    L = len(parameters)//2

    for l in range(1, L):
        cache["Z" + str(l)] = tf.matmul(parameters["W" + str(l)], activations["A" + str(l - 1)]) + parameters["b" + str(l)]
        if(activation_func == "sigmoid"):
            activations["A" + str(l)] = tf.sigmoid(cache["Z" + str(l)])
        elif(activation_func == "relu"):
            activations["A" + str(l)] = tf.nn.relu(cache["Z" + str(l)])
        elif(activation_func == "tanh"):
            activations["A" + str(l)] = tf.tanh(cache["Z" + str(l)])
        else:
            print("Error. Invalid Activation Function.")
        
        ## Dropout Regularization
        #cache["d" + str(l)] = np.random.rand(activations["A" + str(l)].shape[0], activations["A" + str(l)].shape[1])
        #cache["d" + str(l)] = cache["d" + str(l)] < Keep_prob
        #activations["A" + str(l)] = np.multiply(activations["A" + str(l)], cache["d" + str(l)])
        #activations["A" + str(l)] /= Keep_prob
        

    cache["Z" + str(L)] = tf.matmul(parameters["W" + str(L)], activations["A" + str(L - 1)]) + parameters["b" + str(L)]
    #activations["A" + str(L)] = sigmoid(cache["Z" + str(L)])


    ## The Dropout could've been iniatilized by running another for-loop over the activations 
    ##  but in the present way, there's no need of another for-loop

    return cache["Z" + str(L)]


def compute_cost(ZL, Y):    
    # Softmax Loss Function
    ZL = tf.transpose(ZL)
    Y = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ZL, labels = Y))


    return cost

def nn_model(X_train, Y_train, X_test, Y_test, layer_dims = [20,10,5], n_L = 3, optimizer = "adams", activation_func = "sigmoid", lambd = 0.8, learning_rate = 1,
            num_epoch = 10000, mini_batch_size = 256, beta1 = 0.9, beta2 = 0.999, print_cost = False):
    # n_L = Number of Hidden Layers
    m = X_train.shape[1]
    L = n_L + 1
    layer_dims.append(Y_train.shape[0])
    layer_dims = np.hstack([X_train.shape[0], layer_dims])

    print("Number of Layers : " + str(L))
    print("Layer dimensions : " + str(layer_dims))
    
    ops.reset_default_graph()               # Clears the default graph stack and resets the global default graph.   To be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)

    X, Y = create_placeholders(layer_dims[0], layer_dims[-1])
    
    parameters = initialize_parameters(layer_dims)

    ZL = forward_prop(X, parameters, activation_func)

    cost = compute_cost(ZL, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    #if(optimizer == "momentum"):
    #    optimize = tf.train.MomentumOptimizer(learning_rate = learning_rate).minimize(cost)
    #elif(optimizer == "rmsprop"):
    #    optimize = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
    #elif(optimizer == "adams"):
    #    optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #elif(optimizer == "gd"):
    #    optimize = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    #else:
    #    print("Optimizer Algorithm Not Supported")

    seed = 3
    costs = []

    # Initialize all the variables
    init = tf.global_variables_initializer()

        
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_epoch):
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)
            num_mini_batches = int(m/mini_batch_size)
            epoch_cost = 0.

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_mini_batches


            #learning_rate /=  (1 + i)
            if(learning_rate == 0):
                break


            if(i%100 == 0 and override == 0):
                costs.append(epoch_cost)
            if(print_cost):
                print("Cost after Epoch %i : %f  |  Learning_rate : %f " %(i,epoch_cost,learning_rate))

        parameters = sess.run(parameters)
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    
        return parameters, costs, ZL

def predict(parameters, X, activation_func):
    L = len(parameters)//2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = tf.convert_to_tensor(parameters["W" + str(l)])
        parameters["b" + str(l)] = tf.convert_to_tensor(parameters["b" + str(l)])

    x = tf.placeholder(dtype = tf.float32, shape = (X.shape[0], X.shape[1]))

    ZL = forward_prop(x, parameters, activation_func)
    sess = tf.Session()
    

    predictions = sess.run(ZL, feed_dict = {x: X})
    predictions = np.exp(predictions)/np.sum(np.exp(predictions))
    z = np.argmax(predictions, axis=0)
    for i in range(ZL.shape[1]):
        predictions[:,i] = 0
        predictions[z[i],i] = 1
 
    return predictions
 

n_L_default = 3
lambd_default = 0
lr_default = 0.0001
ni_default = 10000
Keep_prob_default = 1
activation_func_default = "relu"
optimizer_default = "adams"
num_epoch_default = 10
mini_batch_size_default = 32
beta1_default = 0.9
beta2_default = 0.999

class NotPositiveError(UserWarning):
	pass
class SmallNumberError(UserWarning):
    pass
class InvalidActivation_Func(UserWarning):
    pass
class InvalidOptimizer(UserWarning):
    pass

if override == False:
    n_L = input("Number of Hidden Layers? \t")
    
    try:
        n_L = int(n_L)
        if n_L <= 0:
            raise NotPositiveError
    except ValueError:
      print("%s is not an integer.\n" %n_L)
      n_L = n_L_default
    except NotPositiveError:
      print("The number is not positive, Using the default Value %i as the Number of Hidden Layers." %n_L_default)
      n_L = n_L_default
    except:
      print("Invalid Input. Using Default Value : %i" %n_L_default)
      n_L = n_L_default

    layer_dims = []
    for i in range(n_L):
        dims = input("Number of Hidden Units for layer %i\t" %(i + 1))
        dims_default = (10*n_L) - (10*i)
        try:
            dims = int(dims)
            if dims <= 0:
                raise NotPositiveError
            layer_dims.append(dims)
        except ValueError:
          print("%s is not an integer.\n" %dims)
          layer_dims.append(dims_default)
        except NotPositiveError:
          print("The number is not positive, Using the default Value %i as the Number of Hidden units for this layer." %dims_default)
          layer_dims.append(dims_default)
        except:
          print("Invalid Input. Using Default Value : %i" %dims_default)
          layer_dims.append(dims_default)


    print(layer_dims)
    
    try:
        activation_func = input("Activation Function? \t").lower()
        if(activation_func != "sigmoid" and activation_func != "tanh" and activation_func != "relu"):
            raise InvalidActivation_Func
    except InvalidActivation_Func:
        print("Invalid Activation Function. Using Defualt Function : '%s'" %activation_func_default)
        activation_func = activation_func_default
    except:
      print("Invalid Activation Function. Using Defualt Function : '%s'" %activation_func_default)
      activation_func = activation_func_default

    try:
        optimizer = input("Optimizing Function (Momentum | Adam | RMSprop) ? \t").lower()
        if(optimizer != "momentum" and optimizer != "adams" and optimizer != "rmsprop"):
            raise InvalidOptimizer
    except:
        print("Invalid Optimizer Algorithm Chosen. Using Default " + str(optimizer_default) + " Algorithm")
        optimizer = optimizer_default


    lambd = input("Regularization Parameter? \t")
    try:
        lambd = float(lambd)
        if lambd < 0:
            raise NotPositiveError
    except ValueError:
      print("%s is not an integer.\n" %lambd)
      lambd = lambd_default
    except NotPositiveError:
      print("The number is negative, Using the default Value %i as the Number of Hidden Layers." %lambd_default)
      lambd = lambd_default

    Keep_prob = input("Value of Dropout Parameter? \t")
    try:
        Keep_prob = float(Keep_prob)
        if Keep_prob < 0:
            raise NotPositiveError
    except ValueError:
      print("< %s is not an integer.>\n" %Keep_prob)
      Keep_prob = Keep_prob_default
    except NotPositiveError:
      print("The number is not positive, Using the default Value %i as the Number of Hidden Layers." %Keep_prob_default)
      Keep_prob = Keep_prob_default

    lr = input("Learning Rate? \t")
    try:
        lr = float(lr)
        if lr <= 0:
            raise NotPositiveError
    except ValueError:
      print("%s is not a float number.\n" %lr)
      lr = lr_default
    except NotPositiveError:
      print("The number is not positive, Using the default Value %i as Learning_Rate." %lr_default)
      lr = lr_default

    num_epoch = input("Number of Epochs? \t")
    try:
        num_epoch = int(num_epoch)
        if(num_epoch <= 0):
            raise NotPositiveError
    except ValueError:
        print("Input is not an integer.\n")
        num_epoch = num_epoch_default
    except SmallNumberError:
        print("Number is too small. Using default Value %f for Number of Iterations" %num_epoch_default)
        num_epoch = num_epoch_default
    except NotPositiveError:
        print("The number is not positive, Using the default Value %f as Number of Iterations." %num_epoch_default)
        num_epoch = num_epoch_default

    try:
        mini_batch_size = int(input("Mini Batch Size?"))
        if(mini_batch_size <= 0):
            raise NotPositiveError
    except ValueError:
        print("Input is not an integer. \n")
        mini_batch_size = mini_batch_size_default
    except SmallNumberError:
        print("Number is too small. Using default Value %f for Number of Iterations" %mini_batch_size_default)
        mini_batch_size = mini_batch_size_default
    except NotPositiveError:
        print("The number is not positive, Using the default Value %f as Number of Iterations." %mini_batch_size_default)
        mini_batch_size = mini_batch_size_default


    beta1 = input("Beta1 parameter? \t")
    try:
        beta1 = float(beta1)
        if beta1 < 0:
            raise NotPositiveError
    except NotPositiveError:
      print("The number is negative, Using the default Value %f as the Number of Hidden Layers." %beta1_default)
      beta1 = beta1_default
    except:
        print("Invalid Input. Using the default Value Beta1 : %f" %beta1_default)
        beta1 = beta1_default

    beta2 = input("Beta2 parameter? \t")
    try:
        beta2 = float(beta2)
        if beta2 < 0:
            raise NotPositiveError
    except NotPositiveError:
      print("The number is negative, Using the default Value %f as the Number of Hidden Layers." %beta2_default)
      beta2 = beta2_default
    except:
        print("Invalid Input. Using the default Value Beta2 : %f" %beta2_default)
        beta2 = beta2_default

else:
    n_L = n_L_default
    layer_dims = [25, 12]
    lr = lr_default
    lambd = lambd_default
    Keep_prob = Keep_prob_default
    activation_func = activation_func_default
    optimizer = optimizer_default
    num_epoch = num_epoch_default
    mini_batch_size = mini_batch_size_default
    beta1 = beta1_default
    beta2 = beta2_default

print("Number of Hidden Layers : " + str(n_L))
print("Layer Dimensions : " + str(layer_dims))
print("Learning Rate : " + str(lr))
print("Optimizer Algorithm : " + str(optimizer))
print("Number of epochs : " + str(num_epoch))
print("Mini Batch Size :" + str(mini_batch_size))
print("Regularization Parameter (lambda) : " +  str(lambd))
print("Dropout Parameter (Keep_prob) : " + str(Keep_prob))
print("Beta1 : " + str(beta1))
print("Beta2 : " + str(beta2))




#Start Training the Model

print("\n\nTraining The Model")
start_training_time = time.time()
learned_parameters, costs, ZL = nn_model(X, Y, X_test, Y_test, layer_dims, n_L = n_L, optimizer = optimizer, activation_func = activation_func, lambd = lambd, learning_rate = lr, 
                                     num_epoch = num_epoch, mini_batch_size = mini_batch_size, beta1 = beta1, beta2 = beta2, print_cost = True)
end_training_time = time.time()

#End Training the Model



def example_X():
    try:
        # Example of a picture
        sel = np.random.randint(1, X_test.shape[1])
        plt.imshow(X_test[:,sel].reshape((num_px, num_px, 3)))
        plt.title("Original Value  :  " + str(np.argmax(Y_test[:,sel])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[:,sel])))
        plt.show()

        try:
            input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
        except:
            return
        if input1 == "1":
            example_X()
        else:
            return
    except:
        return

def example_S():
    sel = np.random.randint(1, X_test.shape[1])
    plt.imshow(X_test[:,sel].reshape((num_px, num_px, 3)))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[:,sel])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[:,sel])))
    plt.show()

    try:
        input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
    except:
        example_number
    if input1 == "1":
        example_S()
    else:
        return

def example_N():
    sel = np.random.randint(1, X_test.shape[1])
    plt.imshow(X_test[:,sel].reshape(20, 20))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[:,sel])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[:,sel])))
    plt.show()

    try:
        input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
    except:
        return
    if input1 == "1":
        example_N()
    else:
        return

def example_Nb():
    print(X_test.shape)
    sel = np.random.randint(1, X_test.shape[1])
    plt.imshow(X_test[:,sel].reshape(28, 28))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[:,sel])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[:,sel])))
    plt.show()

    try:
        input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
    except:
        return
    if input1 == "1":
        example_Nb()
    else:
        return

if override == 0:
    if dataset_option == "X":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        #print ('Test Accuracy: %d' % float((np.dot(Y_test,Y_prediction_test.T) + np.dot(1-Y_test,1-Y_prediction_test.T))/float(Y_test.size)*100) + '%')
        num_px = train_set_x_orig.shape[1]
        example_X()
    elif dataset_option == "S":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        example_S()
    elif dataset_option == "N":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print(Y_test.shape)
        print(Y_prediction_test.shape)
        print(Y_prediction_test)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        example_N()
    elif dataset_option == "Nb":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print(Y_test.shape)
        print(Y_prediction_test.shape)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        example_Nb()
    else:
        #plot the decision boundary
        plot_decision_boundary(lambda x: predict(learned_parameters, x.T, activation_func =  activation_func), X, Y)
        plt.title("decision boundary for hidden layer size " + str(4))
        plt.show()

    #plot cost vs iterations
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate :" + str(lr) + "\nDataset_option : " + str(dataset_option) + "\nlambd : " + str(lambd) + " Epochs : " + str(num_epoch))
    plt.show()





end_time = time.time()
print("Execution Time : " + str(end_time - start_time) + " sec")
print("Training Time : " + str(end_training_time - start_training_time) + " sec")
