import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 200)

df = pd.read_csv('../iris.data', sep=',', header=None, names=['Sepal_Length','Sepal_Width',
                                                                   'Petal_Length','Petal_Width','Species_Class'])

#Preparing the input data
X = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
X = np.array(X)
#X[:5]

#To map the value of species to numerical we use oneHotencoder 
encoder = OneHotEncoder(sparse=False)

Y = df.Species_Class
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1))
#Y[:5]

#Splitting the data into test, train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

#initilaizing weights
def initializing_Weight(nodes):
    layers, weights = len(nodes), []

    for i in range(1, layers):
        wt = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)] for j in range(nodes[i])]
        weights.append(np.matrix(wt))

    return weights

#Creating neural network
def neural_network(X_train, Y_train, X_val=None, Y_val=None, iterations=10, nodes=[], rate=0.15):
    hiddenLayers = len(nodes) - 1
    weights = initializing_Weight(nodes)

    for iteration in range(1, iterations+1):
        weights = trainNetwork(X_train, Y_train, rate, weights)

        #Print the accuracy of training and validation after every 20 iterations
        if(iteration % 20 == 0):
            print("Iteration {}".format(iteration))
            print("Training Accuracy:{}".format(accuracy(X_train, Y_train, weights)))
            if X_val.any():
                print("Validation Accuracy:{}".format(accuracy(X_val, Y_val, weights)))

    return weights

#creating feed forward for prediciton
def feed_forward(x, weights, layers):
    output, current_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(current_input, weights[j].T))
        output.append(activation)
        current_input = np.append(1, activation) # add the bias = 1

    return output


#creating backword propagation
def backword_propagation(y, output, weights, layers):
    outputFinal = output[-1]
    error = np.matrix(y - outputFinal) #Calculate the error at last output

    #Back propagate the error
    for j in range(layers, 0, -1):
        currOutput = output[j]

        if(j > 1):
            # Add previous output
            prevOutput = np.append(1, output[j-1])
        else:
            prevOutput = output[0]

        delta = np.multiply(error, sigmoidDerivative(currOutput))
        weights[j-1] += rate * np.multiply(delta.T, prevOutput)

        wt = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, wt) # Calculate error for current layer

    return weights

#This will perform forward and backward propagation, the new weights will be returned n the end
def trainNetwork(X, Y, rate, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Add feature vector

        output = feed_forward(x, weights, layers)
        weights = backword_propagation(y, output, weights, layers)

    return weights
#sigmoid activation fucntion is used
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    return np.multiply(x, 1-x)

def predict(item, weights):
    layers = len(weights)
    item = np.append(1, item)

    #forward propagation
    output = feed_forward(item, weights, layers)

    outputFinal = output[-1].A1
    index = MaxActivation(outputFinal)

    y = [0 for i in range(len(outputFinal))]
    y[index] = 1

    return y

def MaxActivation(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i

    return index

def accuracy(X, Y, weights):
    correct_classification = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        prediction = predict(x, weights)

        if(y == prediction):
            correct_classification += 1

    return correct_classification / len(X)

features = len(X[0]) # Number of features
classes = len(Y[0]) # Number of classes

layers = [features, 5, 10, classes]
rate, iterations = 0.15, 100

weights = neural_network(X_train, Y_train, X_val, Y_val, iterations=iterations, nodes=layers, rate=rate)

#accuracy
print("Accuracy: {}".format(accuracy(X_test, Y_test, weights)))

print( list(Y_test[0]))  # prints input data
print(predict(X_test[0], weights)) #prints predicited data
