import numpy as np
import csv
import decimal

#for showing the number in right format
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:.10f}'.format})

#activation function
def linear(x):
    return - x / 200

def linear_derivative(x):
    return - 1 / 200

#pre-proccessing the data
#reading the file
file = open('BankWages.csv')
csvReader = csv.reader(file)

data = []

#change our file to a list
for row in csvReader:
    data.append(row)

#removing the first row a list as though it doesn't count as input
data.pop(0)
#also remove the first index of any row cause it count the number of data and should not be included in our input vector
for row in data:
    row.pop(0)

#for not overflowing select portion of data
data = data[9:18]

#change job columns to have number instead of a name
titles = set()
for row in data:
    titles.add(row[0])
title_values = np.arange(len(titles))
titles_dict = dict(zip(titles, title_values))
for row in data:
    for key in titles_dict:
        if row[0] == key:
            row[0] = titles_dict[key]

#change education alternative to have number started from 0 instead of a large number that it already has in its column
education = set()
for row in data:
    education.add(row[1])
education_values = np.arange(len(education))
education_dict = dict(zip(education, education_values))
for row in data:
    for key in education_dict:
        if row[1] == key:
            row[1] = education_dict[key]

#change columns of gender to have number 0 for female and number 1 for male
#change minority columns to have 0 for no and 1 for yes
for row in data:
    if row[2] == 'female':
        row[2] = 0
    else:
        row[2] = 1
    if row[3] == 'yes':
        row[3] = 1
    else:
        row[3] = 0
    # row[1] = int(row[1]) #change the format of education column to integer


#split our data ot input and output
inputs = []
outputs = []
for row in data:
    inputs.append(row[:3])
    outputs.append(row[3:])
inputs = np.array(inputs)
outputs = np.array(outputs)

#assign random value to our weights
np.random.seed(1)
weights = 2 * np.random.random((3 , 1)) - 1
print(f'Initial weights:\n{weights}')

#building the network
for iteration in range(2000):

    input_layer = inputs
    weighted_sum = np.dot(inputs,weights)
    output = linear(weighted_sum)

    #calculating the error
    error = np.subtract(outputs,output)

    adjustments = error * linear_derivative(output)
    
    #update the weights
    weights += np.dot(inputs.T, adjustments)

print(f'Weights after training:\n{weights}')
print(f'Outputs after training:\n{output}')

