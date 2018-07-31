import numpy as np
from data_prep import features, targets, features_test, targets_test

# Define sigmoid function
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

# define sigmoid prime, the differentiation of sigmoid
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

# use to same seed to make debugging easier

np.random.seed(42)

n_records, n_features = features.shape
last_loss = None


# Neural network hyper parameters
n_hidden = 2
epochs = 1000
learnrate = 0.05

#initialize weights
#we'll initialize the weights from a normal distribution centered at 0. A good value for the scale is
# 1/n^0.5 where n is the number of input units. This keeps the input to the sigmoid low for increasing
# numbers of input units.

#weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

weights_input_hidden = np.random.normal(scale=1 / n_features**.5, size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1/ n_features**.5, size=n_hidden)






for e in range(epochs):
    #del_w = np.zeros(weights.shape)
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    #Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        #calculating the output
        #output = sigmoid(np.dot(x, weights))
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

        #calculate the error
        error =  y - output


        #backprop error term for output unit is

        BackErrorOutput = error * output * (1 - output)

        hidden_error = np.dot(BackErrorOutput, weights_hidden_output)


        #backprop error term for hidden unit is

        BackErrorHidden =  hidden_error * hidden_output * (1-hidden_output)

        #the gradient descent step, the error times the gradient times the inputs
        #del_w += error * output * (1-output) * x
        #del_w += error * sigmoid_prime(x) * x
        del_w_hidden_output += BackErrorOutput * hidden_output
        del_w_input_hidden += BackErrorHidden * x[:, None]

    #Update weights
    #weights += learnrate * del_w / n_records

    weights_input_hidden = learnrate * del_w_input_hidden / n_records
    weights_hidden_output = learnrate * del_w_hidden_output / n_records

    #Printing out the mean square error on the training set
    if e % (epochs/ 10)== 0:
        #out = sigmoid(np.dot(features, weights))
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss =np.mean((out - targets)**2)
        if last_loss and last_loss < loss:
            print("Train loss:", loss, " WARNING - Loss Increasing")
        else:
            print("Train loss:", loss)
        last_loss = loss


# Calculate accuracy on test data

#test_out = sigmoid(np.dot(features_test, weights))


hidden = sigmoid(np.dot(features_test,weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions==targets_test)
print("Prediction accuracy: {:.3f}". format(accuracy))





