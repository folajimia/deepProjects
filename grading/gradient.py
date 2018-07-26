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


#initialize weights
#we'll initialize the weights from a normal distribution centered at 0. A good value for the scale is
# 1/n^0.5 where n is the number of input units. This keeps the input to the sigmoid low for increasing
# numbers of input units.

weights = np.random.normal(scale=1 / n_features**.5, size=n_features)


# Neural network hyper parameters
epochs = 1000
learnrate = 0.5



for e in range(epochs):
    del_w = np.zeros(weights.shape)
    #Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        #calculating the output
        output = sigmoid(np.dot(x, weights))

        #calculate the error
        error =  y - output

        #the gradient descent step, the error times the gradient times the inputs
        #del_w += error * output * (1-output) * x
        del_w += error * sigmoid_prime(x) * x

    #Update weights
    weights += learnrate * del_w / n_records

    #Printing out the mean square error on the training set
    if e % (epochs/ 10)== 0:
        out = sigmoid(np.dot(features, weights))
        loss =np.mean((out - targets)**2)
        if last_loss and last_loss < loss:
            print("Train loss:", loss, " WARNING - Loss Increasing")
        else:
            print("Train loss:", loss)
        last_loss = loss


# Calculate accuracy on test data

test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions==targets_test)
print("Prediction accuracy: {:.3f}". format(accuracy))





