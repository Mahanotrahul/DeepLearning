

# Package imports
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets, load_dataset
import scipy.io as sio





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
        test = sio.loadmat('datasets/Digit_Classification.mat')
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
        print(X.shape)
        print(X_test.shape)
        #for i in range(Y.shape[1]):
        #    if Y[:,i] == 10:
        #        Y[:,i] = 0
        #print(np.amax(Y))
        #new_rows = np.zeros((np.amax(Y), Y.shape[1]))
        #Y = np.vstack([Y, new_rows])
        #print(Y.shape)

        #for i in range(Y.shape[1]):
        #    c = i//500
        #    Y[c,i] = 1
        #    Y[0,i] = 0
        #    Y[np.argmax(Y[:,i]), i] = 1
        #sio.savemat("datasets/Digit_Classification.mat", {"X": X, "Y": Y})
        Y_train = Y[:,sel[0:set_divide]]
        Y_test = Y[:,sel[set_divide:Y.shape[1]]]
        Y = Y_train
        print(Y.shape)
        print(Y_test.shape)


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
        except:
            index = 100
            print("Wrong Index Number")
        print("Selected Index Number : %i" %index)
        print(str(train_set_y[:,index]) + "It's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture. ")
        plt.imshow(train_set_x_orig[index])
        plt.show()
        
    elif dataset_option == "N":
        sel = np.random.randint(1, X.shape[1])
        plt.imshow(X[:,sel].reshape(20, 20), cmap = 'gray_r')
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


def sigmoid(z):
    x = 1/(1 + np.exp(-z))
    return x

def sigmoid_backward(A):
    s = A*(1 - A)

    return s

def tanh_backward(A1):
    s = 1 - np.power(A1,2)
    return s

def layer_sizes(X, Y, n_h):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_h, n_y

def initialize_parameters(X, Y, n_h):
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y, n_h)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}

    layer_size = {"n_x" : n_x,
                  "n_h" : n_h,
                  "n_y" : n_y}

    return parameters, layer_size

def forward_propagation(parameters, X):
    m = X.shape[1]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1          # size of Z1 = n_h, m
    if override == 0 and dataset_option == "X":
        A1 = sigmoid(Z1)                # size of A1 = n_h, m
    else:
        A1 = np.tanh(Z1)                # size of A1 = n_h, m
    Z2 = np.dot(W2,A1) + b2         # size of Z2 = n_y, m
    A2 = sigmoid(Z2)                # size of A2 = n_y, m

    params = {"Z1" : Z1,
              "A1" : A1,
              "Z2" : Z2,
              "A2" : A2}

    return A2, params

def compute_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y,np.log(1 - A2)))/(Y.shape[1])
    cost = np.squeeze(cost)            # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17 
    return cost


def backward_propagation(params, parameters, X, Y):
    m = X.shape[1]

    A1 = params["A1"]
    A2 = params["A2"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]   

    dZ2 = A2 - Y
    dW2 = (np.dot(dZ2, A1.T))/m
    db2 = (np.sum(dZ2, axis = 1, keepdims= True))/m
    if override == 0 and dataset_option == "X":
        dZ1 = np.multiply(np.dot(W2.T,dZ2), sigmoid_backward(A1))
    else:
        dZ1 = np.multiply(np.dot(W2.T,dZ2), tanh_backward(A1))
    dW1 = (np.dot(dZ1, X.T))/m
    db1 = (np.sum(dZ1, axis = 1, keepdims = True))/m

    grads = {"dW2" : dW2,
             "db2" : db2,
             "dW1" : dW1,
             "db1" : db1}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]


    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2}

    return parameters

def nn_model(X, Y, n_h = 4, learning_rate = 1.2, num_iterations = 2000, print_cost = False):
    #print("\n\n\t\tLearning Rate : " + str(learning_rate))
    #print("\t\tNumber of Iterations : " + str(num_iterations))
    #print("\t\tNumber of Hidden Layers : " + str(n_h) + "\n\n")
    #time.sleep(1)
    parameters, layer_size = initialize_parameters(X, Y, n_h)
    #print(layer_size)
    #print(parameters)
    #time.sleep(1)


    costs = []
    for i in range(num_iterations):
        A2, params = forward_propagation(parameters, X)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, parameters, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate )

        if i%100 == 0 and override == 0:
            costs.append(cost)

        if print_cost and i%100 == 0:
            print("cost after iteration %i : %f" %(i,cost))

    return parameters, costs

def predict(parameters, X):
    A2, params = forward_propagation(parameters, X)
    n_y = A2.shape[0]
    z = np.argmax(A2, axis=0)
    for i in range(A2.shape[1]):
        if A2[z[i],i]>0.5:
            A2[:,i] = 0
            A2[z[i],i] = 1
        else:
            A2[:,i] = 0
    return A2

def nn_model_testCase():
    X_assess, Y_assess = nn_model_test_case()
    parameters, costs = nn_model(X_assess, Y_assess, 4, num_iterations=10000, learning_rate = 1.2, print_cost=True)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    # Print accuracy
    Y = Y_assess
    X = X_assess
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

    predictions = predict(parameters, X_assess)
    print("predictions mean = " + str(np.mean(predictions)))
    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    
#nn_model_testCase()



class NotPositiveError(UserWarning):
	pass
class SmallNumberError(UserWarning):
    pass
## Build a model with a n_h-dimensional hidden layer
if override == False:
    lr = input("Learning Rate? \t")
    try:
        lr = float(lr)
        if lr <= 0:
            raise NotPositiveError
    except ValueError:
      print("%s is not a float number.\n" % lr)
      lr = 1.0
    except NotPositiveError:
      print("The number is not positive, Using the default Value 0.5 as Learning_Rate.")
      lr = 1.0
    ni = input("Number of Iterations? (Default vaue is 10000)\t ")
    try:
        ni = int(ni)
        if ni <= 0:
            raise NotPositiveError
        if ni < 100 and ni > 0:
            raise SmallNumberError
    except ValueError:
        print("Input is not an integer number.\n")
        ni = 10000
    except SmallNumberError:
        print("Number is too small. Using default Value 10000 for Number of Iterations")
        ni = 10000
    except NotPositiveError:
        print("The number is not positive, Using the default Value 10000 as Number of Iterations.")
        ni = 10000
   
    n_h = input("Number of Hidden Layers? (Default Vaue: 3)\t")
    try:
        n_h = int(n_h)
        if n_h < 1:
            print("Invalid Input. Using Defaut Value: 3")
            n_h = 3
    except:
        n_h = 3

else:
    lr = 1.2
    ni = 1000
    n_h = 8
print("Learning Rate : " + str(lr))
print("Number of Iterations : " + str(ni))




#Start Training the Model

print("\n\nTraining The Model")
start_training_time = time.time()
learned_parameters, costs = nn_model(X, Y, n_h = n_h, num_iterations = ni, learning_rate = lr, print_cost=True)
end_training_time = time.time()


## Print accuracy
Y_prediction_train = predict(learned_parameters, X)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))
#print ('Training Accuracy: %d' % float((np.dot(Y,Y_prediction_train.T) + np.dot(1-Y,1-Y_prediction_train.T))/float(Y.size)*100) + '%')


def example_number(Y_prediction_test):
    sel = np.random.randint(1, X_test.shape[1])
    plt.imshow(X_test[:,sel].reshape(20, 20))
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
        Y_prediction_test  =  predict(learned_parameters, X_test)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        print ('Test Accuracy: %d' % float((np.dot(Y_test,Y_prediction_test.T) + np.dot(1-Y_test,1-Y_prediction_test.T))/float(Y_test.size)*100) + '%')
    elif dataset_option == "N":
        Y_prediction_test  =  predict(learned_parameters, X_test)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        example_number(Y_prediction_test)
    else:
        #plot the decision boundary
        plot_decision_boundary(lambda x: predict(learned_parameters, x.T), X, Y)
        plt.title("decision boundary for hidden layer size " + str(4))
        plt.show()

    #plot cost vs iterations
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()


def example_test(num_px):
    try:
        # Example of a picture that was wrongly classified.
        index = int(input("Index of Picture: "))
        print(Y_prediction_test[0,index])
        plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
        print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(Y_prediction_test[0,index])].decode("utf-8") +  "\" picture.")
        plt.show()
        example_test(num_px)
    except:
        print("Try Again")
        example_test(num_px)
if override == 0:
    if dataset_option == "X":
        num_px = train_set_x_orig.shape[1]
        example_test(num_px)


end_time = time.time()
print("Execution Time : " + str(end_time - start_time) + " sec")
print("Training Time : " + str(end_training_time - start_training_time) + " sec")
