from Math import Matrix
from NeuralNetwork import NeuralNetwork
import pandas
import numpy as np

Testing = True
Import = True if Testing else False
Export = False if Testing else True

NumberOfHiddenLayers = 3  # minimum 1
InputSize = 784  # input size 784
LayerSizes = [12, 12, 10]  # N layer sizes
OutputSize = 10
LearningRate = 0.75

BatchSize = 100

ImageNN = NeuralNetwork(InputSize, OutputSize,
                        NumberOfHiddenLayers, LayerSizes, LearningRate)
ImageNN.build()

if Import == False:
    ImageNN.generate()

else:
    ImageNN.IMPORT_DATA('./NeuralNetworkData/')

if Testing == False:
    TrainingData = pandas.read_csv('mnist_train.csv')
else:
    TrainingData = pandas.read_csv('mnist_test.csv')

TrainingDataImageList = TrainingData.values.tolist()


if Testing == False:
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
        print(Batch, ' ', BatchAverageCost)

        ImageNN.backward_propagation(BatchAverageInfluence)

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
