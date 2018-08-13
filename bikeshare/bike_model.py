import numpy as np
import sys

from data_prep import features, targets, test_features, test_targets,train_features, train_targets,val_features, val_targets




class NeuralNetwork(object):
    def __init__(self, input_nodes, output_nodes, hidden_nodes, learning_rate):

        # set the number of nodes in input, hidden and output layers
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes


        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,(self.output_nodes, self.hidden_nodes))


        # learning rate

        self.lr = learning_rate

        # set activation function as a sigmoid function

        self.activation = lambda x: 1/(1 + np.exp(-x))

    def train(self, input_list, target_list):
        #convert input list to 2D array
        inputs = np.array(input_list, ndim=2).T
        targets = np.array(target_list, ndim=2).T


        # implementing the forward pass

        # hidden layers

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) #input to hidden layer

        hidden_outputs = np.dot(self.activation_function(hidden_inputs)) #hidden layer out activation fn

        #output layer
        final_outputs = np.dot(self.weights_hidden_to_output, hidden_outputs) #signals into final output layer

        final_outputs = final_inputs

        ### implement the backward pass  ###

        # Output error
        # error = y-y' = targets - final_outputs
        output_errors = (targets - final_outputs) * 1 # Output layer error is the difference between desired
        # target and actual output since outer layer input equals output

        #Back propergated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors) #errors propergated to the hidden layer
        hidden_grad = hidden_outputs * (1-hidden_outputs) #hidden layer gradient


        #update weights
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs.T
        self.weights_input_to_hidden += self.lr * hidden_errors * hidden_grad * inputs.T

    def run(self, input_list):
        #Run a forward pass through the network
        inputs = np.array(input_list, ndmin = 2).T


        #The forward pass#

        #Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #output_layer

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs

        return (final_outputs)

    def MSE(y,Y):
        return np.mean((y - Y)**2)


# Set the hyperparameters here #
epochs = 850
learning_rate = 0.2
hidden_nodes = 25
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # go through a batch of 128 records from the training set
    batch








