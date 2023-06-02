from Math import Matrix
from NeuralNetwork import *
import pandas
import random
import numpy as np


NumberOfHiddenLayers = 2  # minimum 1

InputSize = 784  # input size 784
LayerSizes = [10, 8]  # N layer sizes
OutputSize = 10

LearningRate = 0.75


Input = Matrix(InputSize, 1)
HiddenLayers = [Matrix(LayerSizes[layer], 1)
                for layer in range(NumberOfHiddenLayers)]
Output = Matrix(OutputSize, 1)

# Input handling

TrainingData = pandas.read_csv('mnist_train.csv')
TrainingDataImageList = TrainingData.values.tolist()

OneImage = TrainingDataImageList[0]
Label = OneImage[0]
ImagePixels = OneImage[1:]

Temp = list(map(lambda x: round(x/255, 5), ImagePixels))
ImageMatrix = Matrix(1, 784)
ImageMatrix.set([Temp])


expect = [0 for i in range(OutputSize)]
expect[Label] = 1

Input = ImageMatrix.transpose()


'''
StartWeight1 = [[round(random.random(), 5)
                 for x in range(InputSize)] for y in range(LayerSizes)]
StartBias1 = [[round(random.random(), 5)] for y in range(LayerSizes)]

StartWeight2 = [[round(random.random(), 5)
                 for x in range(LayerSizes)] for y in range(OutputSize)]
StartBias2 = [[round(random.random(), 5)] for y in range(OutputSize)]
'''

# Generating random weights
# x is for first and y is for second, so in the Matrices it should be reversed
StartingInputWeights = [[round(random.random(), 5)
                         for x in range(InputSize)] for y in range(LayerSizes[0])]
StartingHiddenLayersWeights = [[[round(random.random(), 5)
                               for x in range(LayerSizes[BetweenHiddenLayers])] for y in range(LayerSizes[BetweenHiddenLayers+1])] for BetweenHiddenLayers in range(NumberOfHiddenLayers - 1)]
StartingOutputWeights = [[round(random.random(), 5)
                          for x in range(LayerSizes[-1])] for y in range(OutputSize)]

# Generating random biases
StartingInputBiases = [[round(random.random(), 5)]
                       for y in range(LayerSizes[0])]
StartingHiddenLayersBiases = [[[round(random.random(), 5)]
                               for y in range(LayerSizes[BetweenHiddenLayers+1])] for BetweenHiddenLayers in range(NumberOfHiddenLayers - 1)]
StartingOutputBiases = [[round(random.random(), 5)]
                        for y in range(OutputSize)]


# Weights and Biases are owned by the first layers, except for the last that are owned by the output layer
# So the last hidden layer "has no weights" beacouse its owned by the output layer

# Initializing Weights in Matrices
InputWeights = Matrix(LayerSizes[0], InputSize)
InputWeights.set(StartingInputWeights)

HiddenLayersWeights = [Matrix(LayerSizes[layer + 1], LayerSizes[layer])
                       for layer in range(NumberOfHiddenLayers-1)]
for layer in range(NumberOfHiddenLayers-1):
    HiddenLayersWeights[layer].set(StartingHiddenLayersWeights[layer])

OutputWeights = Matrix(OutputSize, LayerSizes[-1])
OutputWeights.set(StartingOutputWeights)

# Initializing Biases in Matrices
InputBiases = Matrix(LayerSizes[0], 1)
InputBiases.set(StartingInputBiases)

HiddenLayersBiases = [Matrix(LayerSizes[layer + 1], 1)
                      for layer in range(NumberOfHiddenLayers-1)]
for layer in range(NumberOfHiddenLayers-1):
    HiddenLayersBiases[layer].set(StartingHiddenLayersBiases[layer])

OutputBiases = Matrix(OutputSize, 1)
OutputBiases.set(StartingOutputBiases)


def forward_propagation():
    global Hidden1, Input, Output

    # Hidden1 = Weight1.dot(Input)
    HiddenLayers[0] = InputWeights.dot(Input)
    # Hidden1 = Hidden1+Bias1
    HiddenLayers[0] = HiddenLayers[0] + InputBiases
    # Hidden1 = Hidden1.sigm()
    HiddenLayers[0] = HiddenLayers[0].sigm()

    for layer in range(NumberOfHiddenLayers - 1):
        HiddenLayers[layer+1] = HiddenLayersWeights[layer].dot(
            HiddenLayers[layer])
        HiddenLayers[layer+1] = HiddenLayers[layer+1] + \
            HiddenLayersBiases[layer]
        HiddenLayers[layer+1] = HiddenLayers[layer+1].sigm()

    # Output = Weight2.dot(Hidden1)
    Output = OutputWeights.dot(HiddenLayers[-1])
    # Output = Output+Bias2
    Output = Output + OutputBiases
    # Output = Output.sigm()
    Output = Output.sigm()


def cost_function(output, expected):
    StartListBackProp = []
    Sum = 0
    for i in range(len(output)):
        Sum += (output[i][0]-expected[i])**2
        StartListBackProp.append([2 * (output[i][0]-expected[i])])
    Mx = Matrix(OutputSize, 1)
    Mx.set(StartListBackProp)
    return Sum, Mx


def backward_propagation(UpdatedLayerInfluenceBackwards):
    # UpdatedLayerInfluenceBackwards is at the start the influence that the Output layer has on the Cost
    # and its going through all the neuron layers
    global OutputBiases, OutputWeights, InputBiases, InputWeights, HiddenLayersBiases, HiddenLayersWeights

    # Bias2Gradient = Output.before_sigm().der_of_sigm().mult_same_size(UpdatedLayerInfluenceBackwards)
    OutputBiasesGradient = Output.before_sigm().der_of_sigm(
    ).mult_same_size(UpdatedLayerInfluenceBackwards)
    # Here the before_sigmation layer should be on the same level as the UpdatedLayerInfluenceBackwards

    # Bias2 = Bias2 - Bias2Gradient*LearningRate
    OutputBiases = OutputBiases - OutputBiasesGradient*LearningRate

    # Weight2Gradient = Bias2Gradient.dot(Hidden1.transpose())
    OutputWeightsGradient = OutputBiasesGradient.dot(
        HiddenLayers[-1].transpose())

    # Weight2 = Weight2 - Weight2Gradient*LearningRate  # first updated
    OutputWeights = OutputWeights - OutputWeightsGradient*LearningRate

    # UpdatedLayerInfluenceBackwards = Hidden1 + Weight2Gradient.transpose().sum_right()
    UpdatedLayerInfluenceBackwards = HiddenLayers[-1] + \
        OutputWeightsGradient.transpose().sum_right()

    for layer in range(NumberOfHiddenLayers-1, 0, -1):
        assert (layer != 0)
        HiddenBiasesGradient = HiddenLayers[layer].before_sigm().der_of_sigm(
        ).mult_same_size(UpdatedLayerInfluenceBackwards)

        HiddenLayersBiases[layer-1] = HiddenLayersBiases[layer -
                                                         1] - HiddenBiasesGradient*LearningRate

        # Weight2Gradient = Bias2Gradient.dot(Hidden1.transpose())
        # OutputWeightsGradient = OutputBiasesGradient.dot(
        # HiddenLayers[-1].transpose())
        HiddenWeightsGradient = HiddenBiasesGradient.dot(
            HiddenLayers[layer-1].transpose())

        HiddenLayersWeights[layer-1] = HiddenLayersWeights[layer -
                                                           1] - HiddenWeightsGradient*LearningRate

        UpdatedLayerInfluenceBackwards = HiddenLayers[layer-1] + \
            HiddenWeightsGradient.transpose().sum_right()

    # Bias1Gradient = Hidden1.before_sigm().der_of_sigm().mult_same_size(UpdatedLayerInfluenceBackwards)
    InputBiasesGradient = HiddenLayers[0].before_sigm().der_of_sigm().mult_same_size(
        UpdatedLayerInfluenceBackwards)

    # Bias1 = Bias1 - Bias1Gradient*LearningRate
    InputBiases = InputBiases - InputBiasesGradient*LearningRate

    # Weight1Gradient = Bias1Gradient.dot(Input.transpose())
    InputWeightsGradient = InputBiasesGradient.dot(Input.transpose())

    # Weight1 = Weight1 - Weight1Gradient*LearningRate  # second updated
    InputWeights = InputWeights - InputWeightsGradient*LearningRate

    UpdatedLayerInfluenceBackwards = Input + \
        InputWeightsGradient.transpose().sum_right()


def cycle():
    forward_propagation()
    Cost, StartBackProp = cost_function(Output.get(), expect)
    print(Cost)

    backward_propagation(StartBackProp)


for i in range(5):
    cycle()
