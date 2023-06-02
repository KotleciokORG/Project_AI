from Math import Matrix
import random
import numpy as np


class NeuralNetwork:
    def __init__(self, InputSize, OutputSize, NumberOfHiddenLayers, LayerSizes, LearningRate):
        assert (NumberOfHiddenLayers >= 1)
        self.NumberOfHiddenLayers = NumberOfHiddenLayers  # minimum 1

        self.InputSize = InputSize  # input size 784
        self.LayerSizes = LayerSizes  # N layer sizes
        self.OutputSize = OutputSize

        self.LearningRate = LearningRate

    def build(self):
        self.Input = Matrix(self.InputSize, 1)
        self.HiddenLayers = [Matrix(self.LayerSizes[layer], 1)
                             for layer in range(self.NumberOfHiddenLayers)]
        self.Output = Matrix(self.OutputSize, 1)

    def inject_input(self, Input, Expectation):
        # Arguments as matrices/lists
        Trans = np.array(Input).T.tolist()

        self.Input.set(Trans)
        self.expect = Expectation

    def generate(self):
        # Generating random weights
        # x is for first and y is for second, so in the Matrices it should be reversed
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

        # Initializing Weights in Matrices
        self.InputWeights = Matrix(self.LayerSizes[0], self.InputSize)
        self.InputWeights.set(StartingInputWeights)

        self.HiddenLayersWeights = [Matrix(self.LayerSizes[layer + 1], self.LayerSizes[layer])
                                    for layer in range(self.NumberOfHiddenLayers-1)]
        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersWeights[layer].set(
                StartingHiddenLayersWeights[layer])

        self.OutputWeights = Matrix(self.OutputSize, self.LayerSizes[-1])
        self.OutputWeights.set(StartingOutputWeights)

        # Initializing Biases in Matrices
        self.InputBiases = Matrix(self.LayerSizes[0], 1)
        self.InputBiases.set(StartingInputBiases)

        self.HiddenLayersBiases = [Matrix(self.LayerSizes[layer + 1], 1)
                                   for layer in range(self.NumberOfHiddenLayers-1)]
        for layer in range(self.NumberOfHiddenLayers-1):
            self.HiddenLayersBiases[layer].set(
                StartingHiddenLayersBiases[layer])

        self.OutputBiases = Matrix(self.OutputSize, 1)
        self.OutputBiases.set(StartingOutputBiases)

    def forward_propagation(self):
        # Hidden1 = Weight1.dot(Input)
        self.HiddenLayers[0] = self.InputWeights.dot(self.Input)
        # Hidden1 = Hidden1+Bias1
        self.HiddenLayers[0] = self.HiddenLayers[0] + self.InputBiases
        # Hidden1 = Hidden1.sigm()
        self.HiddenLayers[0] = self.HiddenLayers[0].sigm()

        for layer in range(self.NumberOfHiddenLayers - 1):
            self.HiddenLayers[layer+1] = self.HiddenLayersWeights[layer].dot(
                self.HiddenLayers[layer])
            self.HiddenLayers[layer+1] = self.HiddenLayers[layer+1] + \
                self.HiddenLayersBiases[layer]
            self.HiddenLayers[layer+1] = self.HiddenLayers[layer+1].sigm()

        # Output = Weight2.dot(Hidden1)
        self.Output = self.OutputWeights.dot(self.HiddenLayers[-1])
        # Output = Output+Bias2
        self.Output = self.Output + self.OutputBiases
        # Output = Output.sigm()
        self.Output = self.Output.sigm()

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

        # Bias2Gradient = Output.before_sigm().der_of_sigm().mult_same_size(UpdatedLayerInfluenceBackwards)
        OutputBiasesGradient = self.Output.before_sigm().der_of_sigm(
        ).mult_same_size(UpdatedLayerInfluenceBackwards)
        # Here the before_sigmation layer should be on the same level as the UpdatedLayerInfluenceBackwards

        # Bias2 = Bias2 - Bias2Gradient*LearningRate
        self.OutputBiases = self.OutputBiases - OutputBiasesGradient*self.LearningRate

        # Weight2Gradient = Bias2Gradient.dot(Hidden1.transpose())
        OutputWeightsGradient = OutputBiasesGradient.dot(
            self.HiddenLayers[-1].transpose())

        # Weight2 = Weight2 - Weight2Gradient*LearningRate  # first updated
        self.OutputWeights = self.OutputWeights - \
            OutputWeightsGradient*self.LearningRate

        # UpdatedLayerInfluenceBackwards = Hidden1 + Weight2Gradient.transpose().sum_right()
        UpdatedLayerInfluenceBackwards = self.HiddenLayers[-1] + \
            OutputWeightsGradient.transpose().sum_right()

        for layer in range(self.NumberOfHiddenLayers-1, 0, -1):
            assert (layer != 0)
            HiddenBiasesGradient = self.HiddenLayers[layer].before_sigm().der_of_sigm(
            ).mult_same_size(UpdatedLayerInfluenceBackwards)

            self.HiddenLayersBiases[layer-1] = self.HiddenLayersBiases[layer -
                                                                       1] - HiddenBiasesGradient*self.LearningRate

            # Weight2Gradient = Bias2Gradient.dot(Hidden1.transpose())
            # OutputWeightsGradient = OutputBiasesGradient.dot(
            # HiddenLayers[-1].transpose())
            HiddenWeightsGradient = HiddenBiasesGradient.dot(
                self.HiddenLayers[layer-1].transpose())

            self.HiddenLayersWeights[layer-1] = self.HiddenLayersWeights[layer -
                                                                         1] - HiddenWeightsGradient*self.LearningRate

            UpdatedLayerInfluenceBackwards = self.HiddenLayers[layer-1] + \
                HiddenWeightsGradient.transpose().sum_right()

        # Bias1Gradient = Hidden1.before_sigm().der_of_sigm().mult_same_size(UpdatedLayerInfluenceBackwards)
        InputBiasesGradient = self.HiddenLayers[0].before_sigm().der_of_sigm().mult_same_size(
            UpdatedLayerInfluenceBackwards)

        # Bias1 = Bias1 - Bias1Gradient*LearningRate
        self.InputBiases = self.InputBiases - InputBiasesGradient*self.LearningRate

        # Weight1Gradient = Bias1Gradient.dot(Input.transpose())
        InputWeightsGradient = InputBiasesGradient.dot(self.Input.transpose())

        # Weight1 = Weight1 - Weight1Gradient*LearningRate  # second updated
        self.InputWeights = self.InputWeights - InputWeightsGradient*self.LearningRate

        UpdatedLayerInfluenceBackwards = self.Input + \
            InputWeightsGradient.transpose().sum_right()

    def cycle(self):
        self.forward_propagation()
        Cost, StartBackProp = self.cost_function(
            self.Output.get(), self.expect)

        self.backward_propagation(StartBackProp)
        return Cost
