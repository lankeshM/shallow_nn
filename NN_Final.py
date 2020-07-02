"""
Created on July 15, 2018

@author: Lankesh

"""
import sys
from math import sqrt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import autoencoder


class NuralNetwork:
   
    def __init__(self, input_size, output_size, n_hidden, hidden_size, learning_rate):
        """
        Creates and initiate a neural network with the given parameters. Assume 
        at least one hidden layer exist.
        :param input_size: # of input features
        :param output_size: # of output classes
        :param n_hidden:# of hidden layers
        :param hidden_size: # of neurons in a hidden layer
        :param learning_rate: # weight adjustment according to the according to the loss
        
        """
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.l_rate = learning_rate
        self.weights = []  # Create an array to hold weight matrices of each layer
        
        # Create and initiate first hidden layer with random weights
        first_hidden_layer = []
        for n in range(hidden_size):
            n_array = []
            for w in range(input_size):
                n_array.append(np.random.random())
            first_hidden_layer.append(n_array)          
        self.weights.append(first_hidden_layer)
 
        # Create rest of the hidden layers and initiate them with random weights
        for l in range(n_hidden-1):
            hidden_layer = []
            for n in range(hidden_size):
                n_array = []
                for w in range(hidden_size):
                    n_array.append(np.random.random())
                hidden_layer.append(n_array)          
            self.weights.append(hidden_layer)
 
        # Create output layer and initiate with random weights
        output_layer = []
        for n in range(output_size):
            n_array = []
            for w in range(hidden_size):
                n_array.append(np.random.random())
            output_layer.append(n_array)          
        self.weights.append(output_layer)

        self.network_length = len(self.weights) # size of weight matrices array
        self.outputs = [] # array to hold outputs of each layer calculated by feedforward function
        self.deltas = [] # array to hold deltas (change in output gradient) calculated by back propagation
        for l in range(self.network_length): # make empty lists
            self.outputs.append([]) # append empty lists for hold outputs of each layer
            self.deltas.append([]) # append empty lists for hold deltas of each layer

    @staticmethod
    def __dot_product(v1, v2):
        """
        Returns the dot product of two vectors
        :param v1:column vector
        :param v2:column vector
        :return:
        """
        if len(v1)!= len(v2):   # check the dimensions of two column vectors
            print ("Length mismatch v1 = %d, v2 = %d!" % (len(v1), len(v2)))
            sys.exit(1)
        res = sum([v1[i]*v2[i] for i in range(len(v1))])
        return res

    @staticmethod
    def __sigmoid(x):
        """
        Returns the Sigmoid value for a given output value
        :param: single output value
        :return: Sigmoid function activation value between
        """
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def __sigmoidDerivative(x):
        """
        Returns the derivative of the Sigmoid function for a given output value
        :param: Single output value
        :return: Derivative of the Sigmoid function
        """
        return x * (1.0 - x)

    def __feed_forward(self, input_features):
        """
        Computes the outputs of each layer
        :param: input_features:
        :return: Update the output of each layer (except input layer)
        """
        inputs = input_features  # starting inputs are input features
        for l in range(self.network_length):
            layer_weights = self.weights[l]
            outputs = []
            for n in range(len(layer_weights)):  # calculate output for each neuron in layer l
                weights = layer_weights[n] # weights of neuron n in layer l 
                res = self.__dot_product(weights, inputs)# take dot product of input vector and weight vector of the neuron
                res = self.__sigmoid(res)  # pass the resulted single value of the dot product to Sigmoid function
                outputs.append(res)
            self.outputs[l] = outputs  # save layer's outputs for back propagation
            inputs = outputs           # outputs becomes inputs for next layer

    def __back_propagation(self, expected_output):
        """
        Computes the change of gradient caused by the prediction error for each layer
        :param: expected_output (true class)
        :return: Weight adjustments for each layer
        """
        # back propagate for output layer
        output_weights = self.weights[-1]  # weight of the output layer (last entry of the weights array)
        outputs = self.outputs[-1]  # outputs of the last layer (last entry of the output array)
        output_errors = []  # empty array for hold output errors of each layer
        output_deltas = []  # empty array for hold change of gradient for each layer
        for n in range(len(output_weights)): # iterate over output layer neurons
            err = expected_output[n] - outputs[n]
            delta = err * self.__sigmoidDerivative(outputs[n])
            output_errors.append(err)
            output_deltas.append(delta)
        self.deltas[-1] = output_deltas  # append output layer's deltas array into network deltas array

        # back propagate for all hidden layers
        # Start with last hidden layer
        # order of the algorithm layers x neurons x neurons
        for l in range(self.network_length-2, -1, -1):  # iterate from last hidden layer to the first hidden layer backwards
            layer_weights = self.weights[l]
            layer_outputs = self.outputs[l]
            next_layer_weights = self.weights[l+1]
            next_layer_deltas = self.deltas[l+1]
            layer_errors = []
            layer_deltas = []
            for n in range(len(layer_weights)):  # iterate over the layer n (neurons)
                err = 0.0
                for nx in range(len(next_layer_weights)):  #iterate over the next layer n (neurons)
                    nx_weight = next_layer_weights[nx][n] # nx-th weight of the n-th neuron in layer l
                    nx_delta = next_layer_deltas[nx]
                    err += nx_weight * nx_delta # error for the n-th neuron in layer l
                output = layer_outputs[n]
                delta = err * self.__sigmoidDerivative(output) # calculate change of derivative for each neuron in layer l
                layer_errors.append(err) # hold errors of each neuron in layer l
                layer_deltas.append(delta) # hold deltas of each neuron in layer l
            self.deltas[l] = layer_deltas # save each layer deltas

    def __update_weights(self, input_features):
        """
        Adjust the weights using deltas calculated by backpropagation function
        :param: input feature values
        :return: adjusted layer outputs
        """
        inputs = input_features
        for l in range(self.network_length): # iterate over each layer
            layer_weights = self.weights[l]
            layer_deltas = self.deltas[l]
            for n in range(len(layer_weights)): # iterate over each neuron
                weights = layer_weights[n]
                delta = layer_deltas[n]
                for w in range(len(weights)): # iterate over weight vector of neuron n
                    weights[w] += self.l_rate * delta * inputs[w] 
            inputs = self.outputs[l]  # input to the next layer is this layer output

    @staticmethod
    def __sqr_error(expected, outputs):
        """
        Returns the square error (expected - output)^2
        :param: Expected output
        :param: Predicted outputs
        :return: Squared error (expected - predicted)^2
        """
        err = 0.0
        for i in range(len(outputs)):
            err += (expected[i] - outputs[i]) ** 2
        return err

    def train(self, X, Y, epochs):
        """
        Trains the neural network using the given training data
        :param X:training data
        :param Y:training data labels (classes)
        :param epochs: # of training iterations
        :return: weights for each layer
        """
        # --testing epochs = 1
        error_array = []
        for epoch in range(epochs):
            sqr_error = 0.0
            for i in range(len(X)):
                features = X[i]
                expected = Y[i]
                self.__feed_forward(features)
                self.__back_propagation(expected)
                self.__update_weights(features)
                final_output = self.outputs[-1]
                sqr_error += self.__sqr_error(expected, final_output)
                # print('Error at end of training set %d is %.6f' % (i, sqr_error))
            rmse = sqrt(sqr_error/len(X))
            error_array.append(sqr_error)
            if epoch % 100 == 0:
                print('Error at end of epoch %d is %.4f, rmse = %.4f' % (epoch, sqr_error, rmse))
        return error_array        

    def predict(self, features):
        """
        Predicts targets (class labels) of the test data
        :param: test data feature values
        :return: predicted labels of the test data
        """
        self.__feed_forward(features)
        final_output = self.outputs[-1]
        max_v = final_output[0] 
        max_i = 0 
        for i in range(len(final_output)):
            if final_output[i] > max_v: 
                max_i = i
                max_v = final_output[i]
        return max_i # Return the index of max element

    def print_network(self):
        """
        :return: Print network
        """
        for l in range(self.network_length):
            layer = self.weights[l]
            print ("Layer %d x %d:" % (len(layer[0]), len(layer)))
            for n in range(len(layer)):
                weights = layer[n]
                for w in range(len(weights)):
                    print ("%8.3f " % (weights[w]))
                print ("")


if __name__ == "__main__":

    # RNN.data_preprocessing()

    np.random.seed(1)

    # Import iris data
    iris = datasets.load_iris()
    X = preprocessing.normalize(iris.data, axis=0, norm='max')
    Y = iris.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    lb = preprocessing.LabelBinarizer()
    lb.fit(Y)
    Y_train = lb.transform(Y_train)

    # Network parameters
    feature_size = len(X_train[0])
    output_size = len(Y_train[0])
    hidden_layers = 2
    hidden_layer_size = 5
    l_rate = 0.5

    net = NuralNetwork(feature_size, output_size, hidden_layers, hidden_layer_size, l_rate)
    print ("Network at the beginning:")
    net.print_network()

    # Train network with training data set
    epochs = 1000
    net.train(X_train, Y_train, epochs)
    print ("Network after training:")
    net.print_network()

    # Make predictions and evaluate
    for i in range(len(X_test)):
        x, y = X_test[i], Y_test[i]
        prediction = net.predict(x)
        if prediction == y:
            print ("Expected %d predicted %d - OK" % (y, prediction))
        else:
            print ("Expected %d predicted %d - NG" % (y, prediction))
