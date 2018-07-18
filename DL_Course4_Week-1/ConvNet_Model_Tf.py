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
        train_set_x = train_set_x_orig
        print(train_set_x.shape)
        print(test_set_x_orig.shape)
        test_set_x = test_set_x_orig
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
        Y = dict["Y"].T
        Y_test = dict["Y_test"].T
        print(Y_test)


        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))


    elif dataset_option == "S":
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset_SIGNS()
        train_set_x = train_set_x_orig
        print(train_set_x.shape)
        print(test_set_x_orig.shape)
        test_set_x = test_set_x_orig
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
        Y = dict['Y'].T
        Y_test= dict["Y_test"].T
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
        X = np.zeros((X_train.shape[1], 20, 20))
        X_t = np.zeros((X_test.shape[1], 20, 20))
        for i in range(X_train.shape[1]):
            X[i, :, :] = X_train[:,i].reshape(20,20).T
        for i in range(X_test.shape[1]):
            X_t[i, :, :] = X_test[:, i].reshape(20, 20).T
        X_train = X
        X_test = X_t




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
        Y_train = Y[:,sel[0:set_divide]].T
        Y_test = Y[:,sel[set_divide:Y.shape[1]]].T
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
    train_set_x = train_set_x_orig
    print(train_set_x.shape)
    print(test_set_x_orig.shape)
    test_set_x = test_set_x_orig
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
    Y = dict['Y'].T
    Y_test= dict["Y_test"].T
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
        sel = np.random.randint(1, X.shape[0])
        plt.imshow(X[sel], cmap = "gray_r")
        plt.title("Number: " + str(np.argmax(Y[sel,:])))
        plt.xlabel(Y[sel,:])
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




def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(dtype=tf.float32, shape = (None, n_H0, n_W0, n_C0), name = "X")
    Y = tf.placeholder(dtype=tf.float32, shape = (None, n_y), name = "Y")

    return X,Y

def initialize_parameters(dataset_option):
    parameters = {}
    if dataset_option == "S":
        parameters["W1"] = tf.get_variable(name = "W1", shape = (4, 4, 3, 8), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        parameters["W2"] = tf.get_variable(name = "W2", shape = (2, 2, 8, 16), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        return parameters
    elif dataset_option == "X":
        parameters["W1"] = tf.get_variable(name = "W1", shape = (4, 4, 3, 8), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        parameters["W2"] = tf.get_variable(name = "W2", shape = (2, 2, 8, 16), dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        return parameters

def forward_prop(X, parameters, dataset_option):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    if(dataset_option == "S"):
        Z1 = tf.nn.conv2d(X, W1, strides = (1, 1, 1, 1), padding = "SAME")
        A1 = tf.nn.relu(Z1)

        P1 = tf.nn.max_pool(A1, ksize = (1, 8, 8, 1), strides = (1, 8, 8, 1), padding = "SAME")

        Z2 = tf.nn.conv2d(P1, W2, (1, 1, 1, 1), padding = "SAME")
        A2 = tf.nn.relu(Z2)

        P2 = tf.nn.max_pool(A2, ksize = (1, 4, 4, 1), strides = (1, 4, 4, 1), padding = "SAME")
        P2 = tf.contrib.layers.flatten(P2)
        Z3 = tf.contrib.layers.fully_connected(P2, num_outputs= 6, activation_fn = None)

        return Z3
    elif(dataset_option == "X"):
        Z1 = tf.nn.conv2d(X, W1, strides = (1, 1, 1, 1), padding = "SAME")
        A1 = tf.nn.relu(Z1)

        P1 = tf.nn.max_pool(A1, ksize = (1, 8, 8, 1), strides = (1, 8, 8, 1), padding = "SAME")

        Z2 = tf.nn.conv2d(P1, W2, (1, 1, 1, 1), padding = "SAME")
        A2 = tf.nn.relu(Z2)

        P2 = tf.nn.max_pool(A2, ksize = (1, 4, 4, 1), strides = (1, 4, 4, 1), padding = "SAME")
        P2 = tf.contrib.layers.flatten(P2)
        Z3 = tf.contrib.layers.fully_connected(P2, num_outputs= 2, activation_fn = None)

        return Z3

def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    return cost

def convnet_model(X_train, Y_train, X_test, Y_test, dataset_option, optimizer = "adams", activation_func = "relu", lambd = 0.0001, learning_rate = 0.0001,
          num_epoch = 100, mini_batch_size = 128, beta1 = 0.9, beta2 = 0.999, print_cost = False):

    ops.reset_default_graph()               # Clears the default graph stack and resets the global default graph.   To be able to rerun the model without overwriting tf variables
    (m, n_H0, n_W0, n_C0) = X_train.shape
    (_, n_y) = Y_train.shape
     
    X, Y = create_placeholder(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters(dataset_option)
    Z3 = forward_prop(X, parameters, dataset_option)
    cost = compute_cost(Z3, Y)

    if(optimizer == "momentum"):
        optimize = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = beta1).minimize(cost)
    elif(optimizer == "rmsprop"):
        optimize = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
    elif(optimizer == "adams"):
        optimize = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    elif(optimizer == "gd"):
        optimize = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    else:
        print("Optimizer Algorithm Not Supported")

    costs = []
    seed = 3

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

                _ , minibatch_cost = sess.run([optimize, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_mini_batches


            #learning_rate /=  (1 + i)
            if(learning_rate == 0):
                break


            if(i%100 == 0 and override == 0):
                costs.append(epoch_cost)
            if(print_cost):
                print("Cost after Epoch %i : %f  |  Learning_rate : %f " %(i,epoch_cost,learning_rate))

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters, costs

lr_default = 0.009
lambd_default = 0.0
Keep_prob_default = 1
activation_func_default = "relu"
optimizer_default = "adams"
num_epoch_default = 100
mini_batch_size_default = 64
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
    lr = lr_default
    lambd = lambd_default
    Keep_prob = Keep_prob_default
    activation_func = activation_func_default
    optimizer = optimizer_default
    num_epoch = num_epoch_default
    mini_batch_size = mini_batch_size_default
    beta1 = beta1_default
    beta2 = beta2_default

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
_, _, parameters, _ = convnet_model(X, Y, X_test, Y_test, dataset_option, optimizer = optimizer, activation_func = activation_func, lambd = lambd, learning_rate = lr, 
                                     num_epoch = num_epoch, mini_batch_size = mini_batch_size, beta1 = beta1, beta2 = beta2, print_cost = True)
end_training_time = time.time()

#End Training the Model




