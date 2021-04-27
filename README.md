# Says One Neuron To Another

Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns

### Data sets 
#### •	Title of the data set: Iris Plants Database
Number of instance: 150                    Number of attributes :5
Iris : The Iris flower data set is a multivariate data set having a set of 150 records under five attributes - petal length, petal width, sepal length, sepal width and             species. 

#### •	Title of the data set: Protein Localization Sites
Number of instance: 336                      Number of attributes : 8 
Ecoli: This data set contains information of Escherichia coli. It is a bacterium of the genus Escherichia that is commonly found in the lower intestine of warm-blooded          organism.

### Implementation

1)Preparing the data.

#### Iris Dataset

![Data prepared](https://github.com/samyak3028/Neural_Network/blob/main/preparing(1).png?raw=true)


#### Protein Localization Sites


![Data prepared](https://github.com/samyak3028/Neural_Network/blob/main/preparing(2).png?raw=true)


2)Using one hot encoding technique to convert categorical value to numerical.

3)Splitting data into training and test set.

4)Implementing neural network.
Neural network consists of the An input layer, hidden layer,An output layer, A set of weights and biases between each layer,An activation function for each hidden layer(sigmoid is used here).

5)The weight of network are randomly assigned between (-1,1) and bias has a constant value of 1.

6)The function initializing_Weight take nodes as input which is a list of integer determining no. of nodes in each layer and the function return multi dimensional array.

7)The main function (neural_network) trains the network by iterating , weights need to continuosly adjusted across each iteration to increase the accuracy and after each iteration network is trained using forward/backward propogation. Input is given to network and output is calculated and based on error of outputweights are updated.

8)The forward propogation function(feed_forward) receive input and computes output by dot product of weight and input . Output of one layer works as input to other and final output is predicted.

9)The backward propagation function(backward_propagation) propagates the error backward so that weights can be updated.

10)The trainNetwork function trains the network using forward and backward propagation and updates the weight.

11)The Activation function determines the output of one node which than work as input to other node. I have used Sigmoid activation function . It takes a value as input and outputs another value between 0 and 1. It is non-linear.

12)The prediction function (predict) receive input and output will be array. The higher value determine most probable class so  MaxActivation function is used to find maximum value output and class is predicited.

13)The accuracy function(accuracy) determins the accuracy of function.

14) Final Prediciton is done.

#### Iris Dataset Output

![Data predicted](https://github.com/samyak3028/Neural_Network/blob/main/iteration_prediction(1).png?raw=true)


#### Protein Localization Sites

![Data iterated](https://github.com/samyak3028/Neural_Network/blob/main/iteration(2).png?raw=true)



![Data iterated](https://github.com/samyak3028/Neural_Network/blob/main/prediction(2).png?raw=true)



### Result
Iris dataset was iterated 100 times and Ecoli was 500 times. Accuracy was incresed as iterations increased.  For iris data training accurac was ~98% and testing accuracy was ~87%, whereas for Ecoli dataset ~87% training accuracy, ~82% testing accuracy.

