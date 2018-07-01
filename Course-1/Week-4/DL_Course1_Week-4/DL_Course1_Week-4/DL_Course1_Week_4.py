import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets, load_dataset
import scipy.io as sio
import testCases_v4 as testCases
from functions import sigmoid, relu, sigmoid_backward, relu_backward, tanh_backward

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
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
if override == 0:
    datasets = {"A": noisy_circles,
                "B": noisy_moons,
                "D": blobs,
                "C": gaussian_quantiles}

    dataset_string = {"A": "noisy_circles",
                      "B": "noisy_moons",
                      "D": "blobs",
                      "C": "gaussian_quantiles"}

    dataset_option = input("Which Dataset you want to run the NN Model?\nA: noisy_circles\nB:noisy_moons\nC:Gausssian_Quantiles\nD: Blobs\nX: Image-Classification\nN: Hand-Written Digit Classification\t")
    if any(dataset_option in string for string in dataset_string):
        print("Dataset Choosen : %s" %dataset_string[dataset_option])
        dataset = dataset_option
        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])
        print(X.shape)
        print(Y.shape)
        # make blobs binary
        if dataset == "D":
            Y = Y%2
    elif dataset_option == "X":
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

        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))

    elif dataset_option == "N":

        test = sio.loadmat('datasets/Digit_Classification-BigDataset.mat')
        X = test['X'][:]
        Y = test['Y'][:]
        X_test = test['X_test'][:]
        Y_test = test['Y_test'][:]

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


        #test = sio.loadmat('datasets/Digit_Classification.mat')
        #X = test['X'][:]
        #Y = test['Y'][:]

        ##X, Y = X.T, Y.T
        #sel = np.arange(X.shape[1])
        #np.random.shuffle(sel)
        #print(sel)
        #print(sel.shape)
        #set_divide = (90*X.shape[1])//100
        #X_train = X[:,sel[0:set_divide]]
        #X_test = X[:,sel[set_divide:X.shape[1]]]
        #X = X_train
        #print(X.shape)
        #print(X_test.shape)
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
        #Y_train = Y[:,sel[0:set_divide]]
        #Y_test = Y[:,sel[set_divide:Y.shape[1]]]
        #Y = Y_train
        #print(Y.shape)
        #print(Y_test.shape)


    else:
        print("Wrong Argument. Using Default dataset gaussian_quantiles")
        dataset = "C"

    
else:
    X, Y = gaussian_quantiles
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    print(X.shape)
    print(Y.shape)
  
    



#Example of a picture
if override == 0:
    if dataset_option == "X":
        try:
            index = int(input("Index of the picture( 0 - 209) ? "))
            print("Selected Index Number : %i" %index)
            print(str(train_set_y[:,index]) + "It's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture. ")
            plt.imshow(train_set_x_orig[index])
            plt.show()

        except Exception as e:
            print(e)
        
        
    elif dataset_option == "N":
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


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    return n_x, n_y

def initialize_parameters(layer_dims):
    np.random.seed(3)

    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l - 1])) * (np.sqrt(2/layer_dims[l-1]))   #*0.01 #He Initialization
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_prop(X, parameters, activation_func, Keep_prob):
    cache= {}
    activations = {}
    activations["A" + str(0)] = X
    L = len(parameters)//2

    for l in range(1, L):
        cache["Z" + str(l)] = np.dot(parameters["W" + str(l)], activations["A" + str(l - 1)]) + parameters["b" + str(l)]
        if(activation_func == "sigmoid"):
            activations["A" + str(l)] = sigmoid(cache["Z" + str(l)])
        elif(activation_func == "relu"):
            activations["A" + str(l)] = relu(cache["Z" + str(l)])
        elif(activation_func == "tanh"):
            activations["A" + str(l)] = np.tanh(cache["Z" + str(l)])
        else:
            print("Error. Invalid Activation Function.")
        
        # Dropout Regularization
        cache["d" + str(l)] = np.random.rand(activations["A" + str(l)].shape[0], activations["A" + str(l)].shape[1])
        cache["d" + str(l)] = cache["d" + str(l)] < Keep_prob
        activations["A" + str(l)] = np.multiply(activations["A" + str(l)], cache["d" + str(l)])
        activations["A" + str(l)] /= Keep_prob
        

    cache["Z" + str(L)] = np.dot(parameters["W" + str(L)], activations["A" + str(L - 1)]) + parameters["b" + str(L)]
    activations["A" + str(L)] = sigmoid(cache["Z" + str(L)])

    
    
    ## The Dropout could've been iniatilized by running another for-loop over the activations 
    ##  but in the present way, there's no need of another for-loop

    return cache, activations

def L2_cost(parameters, lambd, m):
    L = len(parameters)//2
    cost = 0
    for l in range(1,L + 1):
        cost = np.sum(np.square(parameters["W" + str(l)])) + cost

    cost = (lambd/(2*m))*cost
    return cost

def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    cost = (-np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y,np.log(1 - AL)))/m)
    if(lambd != 0):
        cost = cost + L2_cost(parameters, lambd, m)
    cost = np.squeeze(cost)         # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17 

    return cost


def back_prop(X, Y, parameters, activations, cache, activation_func, Keep_prob):
    grads = {}
    L = len(parameters)//2
    m = Y.shape[1]

    AL = activations["A" + str(L)]

    grads["dA" + str(L)] =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))     # Initializing the backpropagation
    grads["dZ" + str(L)] = sigmoid_backward(grads["dA" + str(L)], cache["Z" + str(L)])

    

    for l in reversed(range(1, L + 1)):
        grads["dW" + str(l)] = (np.dot(grads["dZ" + str(l)], activations["A" + str(l - 1)].T)/m) + ((lambd/m)*parameters["W" + str(l)])
        grads["db" + str(l)] = np.sum(grads["dZ" + str(l)], axis=1, keepdims = True)/m
        if(l >= 2):
            grads["dA" + str(l - 1)] = np.dot(parameters["W"+ str(l)].T, grads["dZ" + str(l)])
            if(activation_func == "sigmoid"):
                grads["dZ" + str(l - 1)] = sigmoid_backward(grads["dA" + str(l - 1)], cache["Z" + str(l - 1)])
            elif(activation_func == "relu"):
                grads["dZ" + str(l - 1)] = relu_backward(grads["dA" + str(l - 1)], cache["Z" + str(l - 1)])
            elif(activation_func == "tanh"):
                grads["dZ" + str(l - 1)] = tanh_backward(grads["dA" + str(l - 1)], cache["Z" + str(l - 1)])
        
            grads["dA" + str(l - 1)] = np.multiply(grads["dA" + str(l - 1)], cache["d" + str(l - 1)])
            grads["dA" + str(l - 1)] /= Keep_prob

    return grads


def update_parameters(grads, learning_rate, parameters):
    L = len(parameters)//2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*grads["db" + str(l)]

    return parameters


def nn_model(X, Y, layer_dims = [20,10,5], n_L = 3, activation_func = "sigmoid", lambd = 0.8, Keep_prob = 0.76, learning_rate = 1, num_iterations = 2000, print_cost = False):
    # n_L = Number of Hidden Layers

    L = n_L + 1
    layer_dims.append(Y.shape[0])
    layer_dims = np.hstack([X.shape[0], layer_dims])

    print("Number of Layers : " + str(L))
    print("Layer dimensions : " + str(layer_dims))


    costs = []

    parameters = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        cache, activations = forward_prop(X, parameters, activation_func, Keep_prob)

        cost = compute_cost(activations["A" + str(L)], Y, parameters, lambd)

        grads = back_prop(X, Y, parameters, activations, cache, activation_func, Keep_prob)

        parameters = update_parameters(grads, learning_rate, parameters)


        if(i%100 == 0 and override == 0):
            costs.append(cost)
        if(i%100 == 0 and print_cost):
            print("Cost after Iteration %i : %f" %(i,cost))
              
    return parameters, costs

def predict(parameters, X, activation_func):
    _, activations = forward_prop(X, parameters, activation_func, Keep_prob = 1)
 
    AL = activations["A" + str(len(parameters)//2)]
    print(AL.shape)
    n_y = AL.shape[0]
    z = np.argmax(AL, axis=0)
    for i in range(AL.shape[1]):
        if AL[z[i],i]>0.5:
            AL[:,i] = 0
            AL[z[i],i] = 1
        else:
            AL[:,i] = 0
    return AL


n_L_default = 3
lambd_default = 3
lr_default = 0.5
ni_default = 10000
Keep_prob_default = 0.7
activation_func_default = "sigmoid"

class NotPositiveError(UserWarning):
	pass
class SmallNumberError(UserWarning):
    pass
class InvalidActivation_Func(UserWarning):
    pass
## Build a model with a n_h-dimensional hidden layer
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


    ni = input("Number of Iterations? (Default vaue is 10000)\t ")
    try:
        ni = int(ni)
        if ni <= 0:
            raise NotPositiveError
        if ni < 100 and ni > 0:
            raise SmallNumberError
    except ValueError:
        print("Input is not an integer number.\n")
        ni = ni_default
    except SmallNumberError:
        print("Number is too small. Using default Value %i for Number of Iterations" %ni_default)
        ni = ni_default
    except NotPositiveError:
        print("The number is not positive, Using the default Value %i as Number of Iterations." %ni_default)
        ni = ni_default
   
    

else:
    n_L = n_L_default
    layer_dims = [30, 20, 10]
    ni = ni_default
    lr = lr_default
    lambd = lambd_default
    Keep_prob = Keep_prob_default
    activation_func = activation_func_default

print("Number of Hidden Layers : " + str(n_L))
print("Layer Dimensions : " + str(layer_dims))
print("Learning Rate : " + str(lr))
print("Number of Iterations : " + str(ni))
print("Regularization Parameter (lambda) : " + str(lambd))
print("Dropout Parameter (Keep_prob) : " + str(Keep_prob))




#Start Training the Model

print("\n\nTraining The Model")
start_training_time = time.time()
learned_parameters, costs = nn_model(X, Y, layer_dims, n_L = n_L, activation_func = activation_func, lambd = lambd, Keep_prob = Keep_prob, learning_rate = lr, num_iterations = ni, print_cost = True)
end_training_time = time.time()

#End Training the Model

## Print accuracy
Y_prediction_train = predict(learned_parameters, X, activation_func =  activation_func)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))
#print ('Training Accuracy: %d' % float((np.dot(Y,Y_prediction_train.T) + np.dot(1-Y,1-Y_prediction_train.T))/float(Y.size)*100) + '%')


def example_X(num_px):
    try:
        # Example of a picture that was wrongly classified.
        index = int(input("Index of Picture: "))
        print(Y_prediction_test[0,index])
        plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
        print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(Y_prediction_test[0,index])].decode("utf-8") +  "\" picture.")
        plt.show()
        example_X(num_px)
    except:
        return

def example_number(Y_prediction_test):
    sel = np.random.randint(1, X_test.shape[1])
    plt.imshow(X_test[:,sel].reshape(28, 28))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[:,sel])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[:,sel])))
    plt.show()

    try:
        input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
    except:
        example_number
    if input1 == "1":
        example_number(Y_prediction_test)
    else:
        return

if override == 0:
    if dataset_option == "X":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        print ('Test Accuracy: %d' % float((np.dot(Y_test,Y_prediction_test.T) + np.dot(1-Y_test,1-Y_prediction_test.T))/float(Y_test.size)*100) + '%')
        num_px = train_set_x_orig.shape[1]
        example_X(num_px)
    elif dataset_option == "N":
        Y_prediction_test  =  predict(learned_parameters, X_test, activation_func =  activation_func)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        example_number(Y_prediction_test)
    else:
        #plot the decision boundary
        plot_decision_boundary(lambda x: predict(learned_parameters, x.T, activation_func =  activation_func), X, Y)
        plt.title("decision boundary for hidden layer size " + str(4))
        plt.show()

    #plot cost vs iterations
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate :" + str(lr) + "\nDataset_option : " + str(dataset_option) + "\nlambd : " + str(lambd) + " iter : " + str(ni))
    plt.show()





end_time = time.time()
print("Execution Time : " + str(end_time - start_time) + " sec")
print("Training Time : " + str(end_training_time - start_training_time) + " sec")

