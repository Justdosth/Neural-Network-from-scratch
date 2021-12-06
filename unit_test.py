import numpy as np
import csv
import decimal

#for showing the number in right format
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:.10f}'.format})

#activation function
def linear(x):
    return - x / 200 + 4

def linear_derivative(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#creating the data
inputs = np.array([[0,0,1],
                   [1,1,1],
                   [1,0,1],
                   [0,1,1]])
outputs = np.array([[0,1,1,0]]).T

#assign random value to our weights
np.random.seed(1)
weights = 2 * np.random.random((3 , 1)) - 1
print(f'Initial weights:\n{weights}')

#building the network
for iteration in range(10000):

    input_layer = inputs
    weighted_sum = np.dot(inputs,weights)
    output = sigmoid(weighted_sum)
    # output = linear(weighted_sum)

    #calculating the error
    error = np.subtract(outputs,output)
    adjustments = error * sigmoid_derivative(output)
    # adjustments = error * linear_derivative(output)
    
    # print(inputs.T)
    #update the weights
    weights += np.dot(inputs.T, adjustments)

print(f'Weights after training:\n{weights}')
print(f'Outputs after training:\n{output}')
