import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
import scipy.io as sio

override = bool(input("Override(1 for Yes and 0 for No)?"))
print_cost = np.logical_not(override)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,classes = load_dataset()
#print(classes[0].decode("utf-8"))
#print(np.squeeze(classes))
#print(np.squeeze(train_set_y[:,20]))

test = sio.loadmat('datasets/ex4data1.mat')
X = test['X'][:]
Y = test['y'][:]
X, Y = X.T, Y.T

c = np.amax(train_set_y)
print(c)


#Example of a picture
if(override == 0):
    index = int(input("Index of the picture( 0 - 209) ? "))
    print(index)
    print(str(train_set_y[:,index]) + "It's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + " picture. ")
    plt.imshow(train_set_x_orig[index])
    plt.show()


print(train_set_x_orig.shape)
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
print(train_set_x.shape)
print(test_set_x_orig.shape)
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print(test_set_x.shape)

train_set_x = train_set_x/255
test_set_x = test_set_x/255

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

def sigmoid(z):
    x = 1/(1 + np.exp(-z))
    return x

def initialize_parameters(dim):
    #size of W is number of features x 1 i.e w.shape= (num_px,1)
    w = np.zeros((dim,1))
    b = 0
    return w,b

def calculate_cost_and_grads(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T,X) + b)

    dw = np.dot(X,(A - Y).T)/m
    db = np.sum((A - Y))/m
    cost = -1*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1 - A))))/m
    cost = np.squeeze(cost)

    grads = {"dw" : dw,
             "db" : db}

    return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = calculate_cost_and_grads(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

def gradient_descent(w, b, X, Y, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    

    costs = []
    for i in range(num_iterations):
        grads,cost = calculate_cost_and_grads(w, b, X, Y)  
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if print_cost and i%100 == 0:
            costs.append(cost)
            print("Cost after iteration %i : %f" %(i,cost))
        if i <2:
            print(w)
            print(b)

    params = {"w" : w,
              "b" : b}
    grads = {"dw" : dw,
             "db" : db}
    return params, grads, costs
params, grads, costs = gradient_descent(w, b, X, Y, num_iterations= 100, learning_rate = 0.0009,print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


def predict(w,b,X):
    
    w = w.reshape(X.shape[0], 1)
  

    Y_prediction = np.zeros((1,X.shape[1]))

    A = sigmoid(np.dot(w.T,X) + b)

    Y_prediction = A>0.5
    
    return Y_prediction
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))

def model(X_train,Y_train,X_test,Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w,b = initialize_parameters(X_train.shape[0])
    parameter,grads,costs = gradient_descent(w, b, X_train,Y_train, num_iterations, learning_rate, print_cost)
    w = parameter["w"]
    b = parameter["b"]
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


d = model(train_set_x,train_set_y,test_set_x,test_set_y, c, num_iterations = 2000, learning_rate = 0.009, print_cost = True)

# Example of a picture
index = int(input("Index of Picture: "))
print(d["Y_prediction_test"][0,index])
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
plt.show()


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

