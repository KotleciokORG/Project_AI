from Math import Matrix
import random
import numpy as np
import pandas


class NeuralNetwork:
    def __init__(self, InputSize, OutputSize, NumberOfHiddenLayers, LayerSizes):
        assert (NumberOfHiddenLayers >= 1)
        self.NumberOfHiddenLayers = NumberOfHiddenLayers  # minimum 1

        self.InputSize = InputSize  # input size 784
        self.LayerSizes = LayerSizes  # N layer sizes
        self.OutputSize = OutputSize

    def build(self):
        # Initializing Layers
        self.Input = Matrix(self.InputSize, 1)
        self.HiddenLayers = [Matrix(self.LayerSizes[layer], 1)
                             for layer in range(self.NumberOfHiddenLayers)]
        self.Output = Matrix(self.OutputSize, 1)

        # Initializing Weights
        self.InputWeights = Matrix(self.LayerSizes[0], self.InputSize)
        self.HiddenLayersWeights = [Matrix(self.LayerSizes[layer + 1], self.LayerSizes[layer])
                                    for layer in range(self.NumberOfHiddenLayers-1)]
        self.OutputWeights = Matrix(self.OutputSize, self.LayerSizes[-1])

        # Initializing Biases
        self.InputBiases = Matrix(self.LayerSizes[0], 1)
        self.HiddenLayersBiases = [Matrix(
            self.LayerSizes[layer + 1], 1)for layer in range(self.NumberOfHiddenLayers-1)]
        self.OutputBiases = Matrix(self.OutputSize, 1)

        self.GradientCounter = 0
        # Initializing Weights Gradients
        self.InputWeightsGradients = Matrix(self.LayerSizes[0], self.InputSize)
        self.HiddenLayersWeightsGradients = [Matrix(self.LayerSizes[layer + 1], self.LayerSizes[layer])
                                             for layer in range(self.NumberOfHiddenLayers-1)]
        self.OutputWeightsGradients = Matrix(
            self.OutputSize, self.LayerSizes[-1])

        # Initializing Biases Gradients
        self.InputBiasesGradients = Matrix(self.LayerSizes[0], 1)
        self.HiddenLayersBiasesGradients = [Matrix(
            self.LayerSizes[layer + 1], 1)for layer in range(self.NumberOfHiddenLayers-1)]
        self.OutputBiasesGradients = Matrix(self.OutputSize, 1)

    def inject_input(self, Input, Expectation):
        # Arguments as matrices/lists
        Trans = np.array(Input).T.tolist()

        self.Input.set(Trans)
        self.expect = Expectation

    def generate(self):
        # Generating random weights
        # x is for first Layer and y is for second Layer, so in the Matrices it should be reversed
        StartingInputWeights = [[round(random.random(), 5)
                                for x in range(self.InputSize)] for y in range(self.LayerSizes[0])]
        StartingHiddenLayersWeights = [[[round(random.random(), 5)
                                         for x in range(self.LayerSizes[BetweenHiddenLayers])] for y in range(self.LayerSizes[BetweenHiddenLayers+1])] for BetweenHiddenLayers in range(self.NumberOfHiddenLayers - 1)]
        StartingOutputWeights = [[round(random.random(), 5)
                                  for x in range(self.LayerSizes[-1])] for y in range(self.OutputSize)]

        # Generating random biases
        StartingInputBiases = [[round(random.random(), 5)]
                               for y in range(self.LayerSizes[0])]
        StartingHiddenLayersBiases = [[[round(random.random(), 5)]
                                       for y in range(self.LayerSizes[BetweenHiddenLayers+1])] for BetweenHiddenLayers in range(self.NumberOfHiddenLayers - 1)]
        StartingOutputBiases = [[round(random.random(), 5)]
                                for y in range(self.OutputSize)]

        # Weights and Biases are owned by the first layers, except for the last that are owned by the output layer
        # So the last hidden layer "has no weights" beacouse its owned by the output layer

        # Setting Weights
        self.InputWeights.set(StartingInputWeights)

        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersWeights[layer].set(
                StartingHiddenLayersWeights[layer])

        self.OutputWeights.set(StartingOutputWeights)

        # Setting Biases
        self.InputBiases.set(StartingInputBiases)

        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersBiases[layer].set(
                StartingHiddenLayersBiases[layer])

        self.OutputBiases.set(StartingOutputBiases)

    def forward_propagation(self):
        # Hidden1 = Weight1.dot(Input)
        self.HiddenLayers[0] = self.InputWeights.dot(self.Input)
        # Hidden1 = Hidden1+Bias1
        self.HiddenLayers[0] = self.HiddenLayers[0] + self.InputBiases
        # Hidden1 = Hidden1.activate_function()
        self.HiddenLayers[0] = self.HiddenLayers[0].activate_function()

        for layer in range(self.NumberOfHiddenLayers - 1):
            self.HiddenLayers[layer+1] = self.HiddenLayersWeights[layer].dot(
                self.HiddenLayers[layer])
            self.HiddenLayers[layer+1] = self.HiddenLayers[layer+1] + \
                self.HiddenLayersBiases[layer]
            self.HiddenLayers[layer +
                              1] = self.HiddenLayers[layer+1].activate_function()

        # Output = Weight2.dot(Hidden1)
        self.Output = self.OutputWeights.dot(self.HiddenLayers[-1])
        # Output = Output+Bias2
        self.Output = self.Output + self.OutputBiases
        # Output = Output.activate_function()
        self.Output = self.Output.activate_function()

        return self.Output

    def cost_function(self, output, expected):
        StartListBackProp = []
        Sum = 0
        for i in range(len(output)):
            Sum += (output[i][0]-expected[i])**2
            StartListBackProp.append([2 * (output[i][0]-expected[i])])
        Mx = Matrix(self.OutputSize, 1)
        Mx.set(StartListBackProp)
        return Sum, Mx

    def backward_propagation(self, UpdatedLayerInfluenceBackwards):
        # UpdatedLayerInfluenceBackwards is at the start the influence that the Output layer has on the Cost
        # and its going through all the neuron layers

        OutputBiasesGradient = self.Output.before_activate_function().der_of_activate_function(
        ).mult_same_size(UpdatedLayerInfluenceBackwards)
        # Here the before_sigmation layer should be on the same level as the UpdatedLayerInfluenceBackwards

        self.OutputBiasesGradients = self.OutputBiasesGradients + OutputBiasesGradient

        OutputWeightsGradient = OutputBiasesGradient.dot(
            self.HiddenLayers[-1].transpose())

        self.OutputWeightsGradients = self.OutputWeightsGradients + OutputWeightsGradient

        UpdatedLayerInfluenceBackwards = self.HiddenLayers[-1] + \
            OutputWeightsGradient.transpose().sum_right()

        for layer in range(self.NumberOfHiddenLayers-1, 0, -1):
            assert (layer != 0)
            HiddenBiasesGradient = self.HiddenLayers[layer].before_activate_function().der_of_activate_function(
            ).mult_same_size(UpdatedLayerInfluenceBackwards)

            self.HiddenLayersBiasesGradients[layer-1] = self.HiddenLayersBiasesGradients[layer -
                                                                                         1] + HiddenBiasesGradient

            HiddenWeightsGradient = HiddenBiasesGradient.dot(
                self.HiddenLayers[layer-1].transpose())

            self.HiddenLayersWeightsGradients[layer-1] = self.HiddenLayersWeightsGradients[layer -
                                                                                           1] + HiddenWeightsGradient

            UpdatedLayerInfluenceBackwards = self.HiddenLayers[layer-1] + \
                HiddenWeightsGradient.transpose().sum_right()

        InputBiasesGradient = self.HiddenLayers[0].before_activate_function().der_of_activate_function().mult_same_size(
            UpdatedLayerInfluenceBackwards)

        self.InputBiasesGradients = self.InputBiasesGradients + InputBiasesGradient

        InputWeightsGradient = InputBiasesGradient.dot(self.Input.transpose())

        self.InputWeightsGradients = self.InputWeightsGradients + InputWeightsGradient

        UpdatedLayerInfluenceBackwards = self.Input + \
            InputWeightsGradient.transpose().sum_right()

        self.GradientCounter += 1

    def update_network(self, LearningRate):
        self.InputWeights = self.InputWeights - self.InputWeightsGradients * \
            (1/self.GradientCounter) * LearningRate
        self.OutputWeights = self.OutputWeights - \
            self.OutputWeightsGradients * \
            (1/self.GradientCounter) * LearningRate

        self.InputBiases = self.InputBiases-self.InputBiasesGradients * \
            (1/self.GradientCounter) * LearningRate
        self.OutputBiases = self.OutputBiases - self.OutputBiasesGradients * \
            (1/self.GradientCounter) * LearningRate

        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersWeights[layer] = self.HiddenLayersWeights[layer] - \
                self.HiddenLayersWeightsGradients[layer] * \
                (1/self.GradientCounter) * LearningRate
            self.HiddenLayersBiases[layer] = self.HiddenLayersBiases[layer] - \
                self.HiddenLayersBiasesGradients[layer] * \
                (1/self.GradientCounter) * LearningRate

        self.GradientCounter = 0
        self.InputWeightsGradients.clear()
        self.OutputWeightsGradients.clear()

        self.InputBiasesGradients.clear()
        self.OutputBiasesGradients.clear()

        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersWeightsGradients[layer].clear()
            self.HiddenLayersBiasesGradients[layer].clear()

    def forward_cost(self):
        self.forward_propagation()
        Cost, StartBackProp = self.cost_function(
            self.Output.get(), self.expect)
        return Cost, StartBackProp

    def EXPORT_DATA(self, path):
        dfInputWeights = pandas.DataFrame.from_records(self.InputWeights.get())
        dfInputWeights.to_csv(path + 'InputWeights.csv',
                              index=False)

        for Layer in range(self.NumberOfHiddenLayers - 1):
            dfHiddenLayersWeights = pandas.DataFrame.from_records(
                self.HiddenLayersWeights[Layer].get())
            dfHiddenLayersWeights.to_csv(path + 'HiddenLayersWeights_'+str(Layer)+'.csv',
                                         index=False)

        dfOutputWeights = pandas.DataFrame.from_records(
            self.OutputWeights.get())
        dfOutputWeights.to_csv(path + 'OutputWeights.csv',
                               index=False)

        dfInputBiases = pandas.DataFrame.from_records(self.InputBiases.get())
        dfInputBiases.to_csv(path + 'InputBiases.csv',
                             index=False)

        for Layer in range(self.NumberOfHiddenLayers - 1):
            dfHiddenLayersBiases = pandas.DataFrame.from_records(
                self.HiddenLayersBiases[Layer].get())
            dfHiddenLayersBiases.to_csv(path + 'HiddenLayersBiases_'+str(Layer)+'.csv',
                                        index=False)

        dfOutputBiases = pandas.DataFrame.from_records(
            self.OutputBiases.get())
        dfOutputBiases.to_csv(path + 'OutputBiases.csv',
                              index=False)

    def IMPORT_DATA(self, path):
        dfInputWeights = pandas.read_csv(
            path + 'InputWeights.csv').values.tolist()

        self.InputWeights.set(dfInputWeights)

        for Layer in range(self.NumberOfHiddenLayers - 1):
            dfHiddenLayersWeights = pandas.read_csv(
                path + 'HiddenLayersWeights_'+str(Layer)+'.csv').values.tolist()

            self.HiddenLayersWeights[Layer].set(dfHiddenLayersWeights)

        dfOutputWeights = pandas.read_csv(
            path + 'OutputWeights.csv').values.tolist()

        self.OutputWeights.set(dfOutputWeights)

        dfInputBiases = pandas.read_csv(
            path + 'InputBiases.csv').values.tolist()

        self.InputBiases.set(dfInputBiases)

        for Layer in range(self.NumberOfHiddenLayers - 1):
            dfHiddenLayersBiases = pandas.read_csv(
                path + 'HiddenLayersBiases_'+str(Layer)+'.csv').values.tolist()

            self.HiddenLayersBiases[Layer].set(dfHiddenLayersBiases)

        dfOutputBiases = pandas.read_csv(
            path + 'OutputBiases.csv').values.tolist()

        self.OutputBiases.set(dfOutputBiases)
