from Math import Matrix
from NeuralNetwork import NeuralNetwork
import pandas
import numpy as np

NumberOfHiddenLayers = 2  # minimum 1
InputSize = 784  # input size 784
LayerSizes = [10, 8]  # N layer sizes
OutputSize = 10
LearningRate = 0.75

BatchSize = 100

ImageNN = NeuralNetwork(InputSize, OutputSize,
                        NumberOfHiddenLayers, LayerSizes, LearningRate)
ImageNN.build()
ImageNN.generate()

TrainingData = pandas.read_csv('mnist_train.csv')
TrainingDataImageList = TrainingData.values.tolist()

print(len(TrainingDataImageList))

ImageNN.EXPORT_DATA('./NeuralNetworkData/')

Tester = NeuralNetwork(InputSize, OutputSize,
                       NumberOfHiddenLayers, LayerSizes, LearningRate)

Tester.build()
Tester.IMPORT_DATA('./NeuralNetworkData/')


A = Tester.InputWeights.get()
B = ImageNN.InputWeights.get()
assert (A == B)

'''
for Batch in range(round(len(TrainingDataImageList)/BatchSize)):
    BatchAverageCost = 0
    BatchAverageInfluence = Matrix(OutputSize, 1)
    for img in range(BatchSize):
        PresentImageNumber = Batch*BatchSize + img

        OneImage = TrainingDataImageList[PresentImageNumber]
        Label = OneImage[0]
        ImagePixels = OneImage[1:]

        Input = list(map(lambda x: round(x/255, 5), ImagePixels))
        expect = [0 for i in range(OutputSize)]
        expect[Label] = 1

        ImageNN.inject_input([Input], expect)
        IndividualCost, IndividualInfluence = ImageNN.forward_cost()

        BatchAverageCost = BatchAverageCost + IndividualCost
        BatchAverageInfluence = BatchAverageInfluence + IndividualInfluence

    BatchAverageCost = BatchAverageCost * (1/BatchSize)
    BatchAverageInfluence = BatchAverageInfluence * (1/BatchSize)
    print(BatchAverageCost)

    ImageNN.backward_propagation(BatchAverageInfluence)
'''
