from Math import Matrix
from NeuralNetwork import NeuralNetwork
import pandas
import numpy as np
import random

# Test - 1
# Train - 0
Testing_Training = 0
Import = True if Testing_Training else False
Export = False if Testing_Training else True

NumberOfHiddenLayers = 2  # minimum 1
InputSize = 784  # input size 784
LayerSizes = [16, 16]  # N layer sizes
OutputSize = 10
LearningRate = 0.1

BatchSize = 1
Epochs = 2

ImageNN = NeuralNetwork(InputSize, OutputSize,
                        NumberOfHiddenLayers, LayerSizes)
ImageNN.build()

if Import == False:
    ImageNN.generate()

else:
    ImageNN.IMPORT_DATA('./NeuralNetworkData/')

if Testing_Training == False:
    TrainingData = pandas.read_csv('mnist_train.csv')
else:
    TrainingData = pandas.read_csv('mnist_test.csv')

TrainingDataImageList = TrainingData.values.tolist()[:10000]
NumberOfBatches = len(TrainingDataImageList)//BatchSize


if Testing_Training == False:
    for epoch in range(Epochs):
        random.shuffle(TrainingDataImageList)
        for Batch in range(len(TrainingDataImageList)//BatchSize):
            BatchAverageCost = 0
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
                ImageNN.backward_propagation(IndividualInfluence)

            ImageNN.update_network(LearningRate)
            BatchAverageCost = BatchAverageCost * (1/BatchSize)
            print(Batch, '/', NumberOfBatches, ' ', BatchAverageCost)

        if Export == True:
            ImageNN.EXPORT_DATA('./NeuralNetworkData/')

else:
    Accuracy = 0
    for PresentImageNumber in range(len(TrainingDataImageList)):
        OneImage = TrainingDataImageList[PresentImageNumber]
        Label = OneImage[0]
        ImagePixels = OneImage[1:]

        Input = list(map(lambda x: round(x/255, 5), ImagePixels))
        expect = [0 for i in range(OutputSize)]
        expect[Label] = 1

        ImageNN.inject_input([Input], expect)

        Out = ImageNN.forward_propagation().transpose().get()[0]
        LeadingValue = max(Out)
        if Out.index(LeadingValue) == Label:
            Accuracy += 1
            print('GOOD')
        else:
            print('BAD')
        print(round(Accuracy/(PresentImageNumber+1), 5))
