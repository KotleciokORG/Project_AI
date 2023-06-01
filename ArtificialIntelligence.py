from Math import Matrix
import pandas
import random
import numpy as np

InputSize = 2  # input size 784
L1 = 3  # 1st layer size
OutputSize = 1

X = [[3, 2], [4, 5], [6, 7]]


StartWeight1 = [[round(random.random(), 5)
                 for x in range(InputSize)] for y in range(L1)]
StartBias1 = [[round(random.random(), 5)] for y in range(L1)]

StartWeight2 = [[round(random.random(), 5)
                 for x in range(L1)] for y in range(OutputSize)]
StartBias2 = [[round(random.random(), 5)] for y in range(OutputSize)]


Data = pandas.read_csv('mnist_train.csv')
ListData = Data.values.tolist()
OneInp = ListData[0]

# Inp = [OneInp[0], list(
#    map(lambda x: [round(x[0]/255, 5)], np.array(self.mx).T.tolist(OneInp[1:]) ))]

Inp = [0, [[4], [2]]]

expect = [0 for i in range(OutputSize)]
expect[Inp[0]] = 0.5

StartListBackProp = []

Input = Matrix(InputSize, 1)
Hidden1 = Matrix(L1, 1)
Output = Matrix(OutputSize, 1)

Input.set(Inp[1])

Weight1 = Matrix(L1, InputSize)
Weight1.set(StartWeight1)
Bias1 = Matrix(L1, 1)
Bias1.set(StartBias1)

Weight2 = Matrix(OutputSize, L1)
Weight2.set(StartWeight2)
Bias2 = Matrix(OutputSize, 1)
Bias2.set(StartBias2)


def forward_propagation():
    global Hidden1, Input, Output

    Hidden1 = Weight1.dot(Input)
    Hidden1 = Hidden1+Bias1
    Hidden1 = Hidden1.sigm()

    Output = Weight2.dot(Hidden1)
    Output = Output+Bias2
    Output = Output.sigm()


def cost_function(output, expected):
    global StartListBackProp
    StartListBackProp = []
    Sum = 0
    for i in range(len(output)):
        Sum += (output[i][0]-expected[i])**2
        StartListBackProp.append([2 * (output[i][0]-expected[i])])
    return Sum


def backward_propagation():
    global StartBackProp, Hidden1, Weight1, Weight2, Output

    Weight2Gradient = (Output.before_sigm().der_of_sigm().mult_same_size(StartBackProp)
                       ).dot(Hidden1.transpose())
    Weight2 = Weight2 - Weight2Gradient  # first updated
    StartBackProp = Hidden1 + Weight2Gradient.transpose().sum_right()

    Weight1Gradient = (Hidden1.before_sigm().der_of_sigm().mult_same_size(StartBackProp)
                       ).dot(Input.transpose())
    Weight1 = Weight1 - Weight1Gradient  # second updated
    StartBackProp = Input + Weight1Gradient.transpose().sum_right()


forward_propagation()
Cost = cost_function(Output.get(), expect)
print(Cost)

StartBackProp = Matrix(OutputSize, 1)
StartBackProp.set(StartListBackProp)
backward_propagation()


forward_propagation()
Cost = cost_function(Output.get(), expect)
print(Cost)

StartBackProp = Matrix(OutputSize, 1)
StartBackProp.set(StartListBackProp)
backward_propagation()


forward_propagation()
Cost = cost_function(Output.get(), expect)
print(Cost)

StartBackProp = Matrix(OutputSize, 1)
StartBackProp.set(StartListBackProp)
backward_propagation()


forward_propagation()
Cost = cost_function(Output.get(), expect)
print(Cost)

StartBackProp = Matrix(OutputSize, 1)
StartBackProp.set(StartListBackProp)
backward_propagation()
