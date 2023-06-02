from Math import Matrix
from NeuralNetwork import NeuralNetwork
import pandas
import numpy as np

NumberOfHiddenLayers = 2  # minimum 1

InputSize = 784  # input size 784
LayerSizes = [10, 8]  # N layer sizes
OutputSize = 10

LearningRate = 0.75

ImageNN = NeuralNetwork(InputSize, OutputSize,
                        NumberOfHiddenLayers, LayerSizes, LearningRate)

ImageNN.build()
ImageNN.generate()

TrainingData = pandas.read_csv('mnist_train.csv')
TrainingDataImageList = TrainingData.values.tolist()

OneImage = TrainingDataImageList[0]
Label = OneImage[0]
ImagePixels = OneImage[1:]

Input = list(map(lambda x: round(x/255, 5), ImagePixels))
expect = [0 for i in range(OutputSize)]
expect[Label] = 1

ImageNN.inject_input([Input], expect)

print(ImageNN.cycle())
print(ImageNN.cycle())
print(ImageNN.cycle())
print(ImageNN.cycle())
print(ImageNN.cycle())
