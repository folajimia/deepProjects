import numpy as np
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

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)

        hidden_outputs = np.dot(self.activation_function(hidden_inputs))





