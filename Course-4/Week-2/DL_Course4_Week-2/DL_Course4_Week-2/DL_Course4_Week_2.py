import time
start_time = time.time()
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
#from resnets_utils import *
from functions import sigmoid, relu, sigmoid_backward, relu_backward, tanh_backward, one_hot_encoding, random_mini_batches, normalize
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets, load_dataset, load_dataset_SIGNS, load_FACE_dataset
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


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

    dataset_option = (input("Which Dataset you want to run the ConvNet Model on?\nX: Image-Classification\nS : SIGNS Dataset\nN: Hand-Written Digit Classification" +
                            "\nNb : Hand-Written Digits BigDatset [MNIST Datset]\nH: Happy Face Detection\t"))
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

        print(Y)
        print(Y_test)
        print("Y_test.shape : " + str(Y_test.shape))
        #print("X_test.shape : " + str(X_test.shape))
        dict = one_hot_encoding(dict = {"Y" : Y, "Y_test" : Y_test})
        Y = dict["Y"]
        Y_test = dict["Y_test"]
        print(Y)
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
        
        #print(Y_test)

        # One Hot Encoding
        dict = {'Y' : Y, 
                'Y_test' : Y_test}
        dict = one_hot_encoding(dict)
        Y = dict['Y']
        Y_test= dict["Y_test"]
        del dict
        #print(Y_test)

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
        X = np.zeros((X_train.shape[1], 20, 20, 1))
        X_t = np.zeros((X_test.shape[1], 20, 20, 1))
        for i in range(X_train.shape[1]):
            X[i, :, :, 0] = X_train[:,i].reshape(20,20).T
        for i in range(X_test.shape[1]):
            X_t[i, :, :, 0] = X_test[:, i].reshape(20, 20).T
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
        X_train = test['X'][:]
        Y = test['Y'][:].T
        X_test = test['X_test'][:]
        Y_test = test['Y_test'][:].T

        X = np.zeros((X_train.shape[1], 28, 28, 1))
        X_t = np.zeros((X_test.shape[1], 28, 28, 1))
        for i in range(X_train.shape[1]):
            X[i, :, :, 0] = X_train[:,i].reshape(28,28)
        for i in range(X_test.shape[1]):
            X_t[i, :, :, 0] = X_test[:, i].reshape(28, 28)
        X_train = X
        X_test = X_t


        print("Y.shape : " + str(Y.shape))
        print("X.shape : " + str(X.shape))
        print("Y_test.shape : " + str(Y_test.shape))
        print("X_test.shape : " + str(X_test.shape))
    elif dataset_option == "H":
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_FACE_dataset()

        X_train = X_train_orig/255
        X_test = X_test_orig/255
        X = X_train

        # One Hot Encoding
        dict = {'Y' : Y_train_orig, 
                'Y_test' : Y_test_orig}
        dict = one_hot_encoding(dict)
        Y_train = dict['Y'].T
        Y_test= dict["Y_test"].T
        del dict

        Y = Y_train

        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_test shape: " + str(X_test.shape))
        print ("Y_test shape: " + str(Y_test.shape))

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
            plt.imshow(X[index,:,:,:])
            plt.show()

        except Exception as e:
            print(e)
    
    elif dataset_option == "S":
        try:
            index = int(input("Index of the picture( 0 -  %i )  ? " %X.shape[0]))
            print("Selected Index Number : %i" %index)
            plt.imshow(X[index, :, :, :])
            plt.title("Number: " + str(np.argmax(Y[index, :])))
            plt.xlabel(Y[index, :])
            plt.show()
        except Exception as e:
            print(e)
        
    elif dataset_option == "N":
        sel = np.random.randint(1, X.shape[0])
        plt.imshow(X[sel, : ,: , 0], cmap = "gray_r")
        plt.title("Number: " + str(np.argmax(Y[sel,:])))
        plt.xlabel(Y[sel,:])
        plt.show()
    
    elif dataset_option == "Nb":
        sel = np.random.randint(1, X.shape[0])
        plt.imshow(X[sel, : ,: , 0], cmap = "gray_r")
        plt.title("Number: " + str(np.argmax(Y[sel,:])))
        plt.xlabel(Y[sel,:])
        plt.show()
    elif dataset_option == "H":
        sel = np.random.randint(1, X_train.shape[0])
        plt.imshow(X[sel, : ,: , :])
        plt.title("Happy_Face: " + str(np.argmax(Y_train[sel,:])))
        plt.xlabel(Y[sel,:])
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




def identity_block(X, f, filters, stage, block):
    F1, F2, F3 = filters

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = X

    # First Main Component 
    X = Conv2D(F1, (1, 1), strides = (1, 1), padding = "valid", kernel_initializer = glorot_uniform(seed = 0), name =  conv_name_base + "_2a")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2a")(X)
    X = Activation("relu")(X)

    # Second Main Component
    X = Conv2D(F2, (f, f), strides = (1, 1), padding = "same", kernel_initializer = glorot_uniform(seed = 0), name = conv_name_base + "_2b")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2b")(X)
    X = Activation("relu")(X)

    # Third Main Component
    X = Conv2D(F3, (1, 1), strides = (1 ,1), padding = "valid", kernel_initializer = glorot_uniform(seed = 0), name = conv_name_base +  "_2c")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2c")(X)


    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

#tf.reset_default_graph()
#with tf.Session() as test:
#    np.random.seed(1)
#    A_prev = tf.placeholder("float", [3, 4, 4, 6])
#    X = np.random.randn(3, 4, 4, 6)
#    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
#    test.run(tf.global_variables_initializer())
#    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#    print("out = " + str(out[0][1][1][0]))

def convolutional_block(X, f, filters,  stage, block, s = 2):
    F1, F2, F3 = filters
    X_shortcut = X

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # First Component
    X = Conv2D(F1, (1, 1), strides = (s, s), padding = "valid", kernel_initializer = glorot_uniform(seed = 0), name = conv_name_base + "_2a")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2a")(X)
    X = Activation("relu")(X)

    # Second Component
    X = Conv2D(F2, (f, f), strides = (1, 1), padding = "same", kernel_initializer = glorot_uniform(seed = 0), name = conv_name_base + "_2b")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2b")(X)
    X = Activation("relu")(X)

    # Third Component
    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = "valid", kernel_initializer = glorot_uniform(seed = 0), name = conv_name_base + "_2c")(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + "_2c")(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), padding = "valid", kernel_initializer = glorot_uniform(seed= 0), name = conv_name_base + "1")(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + "1")(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

#tf.reset_default_graph()
#with tf.Session() as test:
#    np.random.seed(1)
#    A_prev = tf.placeholder("float", [3, 4, 4, 6])
#    X = np.random.randn(3, 4, 4, 6)
#    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
#    test.run(tf.global_variables_initializer())
#    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#    print("out = " + str(out[0][1][1][0]))

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    X_input = Input(input_shape)

    X = ZeroPadding2D(padding = (3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), padding = "valid", kernel_initializer = glorot_uniform(seed = 0), name = "conv1")(X)
    X = BatchNormalization(axis = 3, name = "bn_conv1")(X)
    X = Activation("relu")(X) 
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'b')
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'b')
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'd')


    X = AveragePooling2D(pool_size = (1, 1), name = "avg_pool")(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet50")
    return model

model = ResNet50(input_shape = (X.shape[1], X.shape[2], X.shape[3]), classes = Y.shape[1])
#print(model.summary())
print("### Compiling Model. ###")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("### Running Model. ###")
start_training_time = time.time()
model.fit(X, Y, epochs = 1, batch_size = 32)



while True:
    try:
        run_again = int(input("Do you want to run the Model again for 1 epoch more? (1 - Yes | 0 - No) : "))
        if run_again == 1:
            model.fit(X, Y, epochs = 1, batch_size = 32)
        else:
            break
    except Exception as e:
        break

end_training_time = time.time()

print("### Evaluating Model. ###")
preds = model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))
print("\n")
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print("### Model Evaluation Completed. ###")

plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

def example_X():
    try:
        # Example of a picture
        sel = np.random.randint(1, X_test.shape[0])
        x = image.img_to_array(X_test[sel, :, :, :])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        plt.imshow(X_test[sel,:,:,:].reshape((num_px, num_px, 3)))
        plt.title("Original Value  :  " + str(np.argmax(Y_test[sel,:])) + "\n Predicted Value :" + str(np.argmax(model.predict(x))))
        plt.show()

        try:
            input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
        except Exception as e:
            print(e)
            example_X()
        if input1 == "1":
            example_X()
        else:
            return
    except Exception as e:
        print(e)
        return

def example_S():
    sel = np.random.randint(1, X_test.shape[0])
    x = image.img_to_array(X_test[sel, :, :, :])
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    plt.imshow(X_test[sel,:,:,:].reshape((num_px, num_px, 3)))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[sel,:])) + "\n Predicted Value :" + str(np.argmax(model.predict(x))))
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
    plt.imshow(X_test[sel,:,:,0].reshape(20, 20))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[sel,:])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[sel,:])))
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
    plt.imshow(X_test[sel, :, :, 0].reshape(28, 28))
    plt.title("Original Value  :  " + str(np.argmax(Y_test[sel, :])) + "\n Predicted Value :" + str(np.argmax(Y_prediction_test[sel, :])))
    plt.show()

    try:
        input1 = input("Show another random Digit Prediction?(1 - YES | 0 - NO)\t")
    except:
        return
    if input1 == "1":
        example_Nb()
    else:
        return

def example_test():
    if override == 0:
        if dataset_option == "X":
            #Y_prediction_test  =  predict(learned_parameters, X_test, dataset_option)
            #print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
            #print ('Test Accuracy: %d' % float((np.dot(Y_test,Y_prediction_test.T) + np.dot(1-Y_test,1-Y_prediction_test.T))/float(Y_test.size)*100) + '%')
            print("## Example ##")
            num_px = X_test.shape[1]
            example_X()
        elif dataset_option == "S":
            #Y_prediction_test  =  predict(learned_parameters, X_test, dataset_option)
            num_px = X_test.shape[1]
            #print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
            example_S()
        elif dataset_option == "N":
            Y_prediction_test  =  predict(learned_parameters, X_test, dataset_option)
            print(Y_test.shape)
            print(Y_prediction_test.shape)
            print(Y_prediction_test)
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
            example_N()
        elif dataset_option == "Nb":
            Y_prediction_test  =  predict(learned_parameters, X_test, dataset_option)
            print(Y_test.shape)
            print(Y_prediction_test.shape)
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
            example_Nb()
        else:
            #plot the decision boundary
            plot_decision_boundary(lambda x: predict(learned_parameters, x.T, activation_func =  activation_func), X, Y)
            plt.title("decision boundary for hidden layer size " + str(4))
            plt.show()

    ##plot cost vs iterations
    #plt.plot(costs)
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per hundreds)')
    #plt.title("Learning rate :" + str(lr) + "\nDataset_option : " + str(dataset_option) + "\nlambd : " + str(lambd) + " Epochs : " + str(num_epoch))
    #plt.show()
#example_test()

print(model.summary())
end_time = time.time()
print("Execution Time : " + str(end_time - start_time) + " sec")
print("Training Time : " + str(end_training_time - start_training_time) + " sec")

print("\n\n### Loading Trained Model. Please Wait")
if dataset_option == "S":
    model = load_model("..\\..\\..\\..\\..\\datasets\\ResNet50_Model\\ResNet50.h5")
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    example_test()
    print(model.summary())