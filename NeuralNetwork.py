class NeuralNetwork:
    def __init__(self):
        pass

    def forward_propagation(self):
        global Hidden1, Input, Output

        Hidden1 = Weight1.dot(Input)
        Hidden1 = Hidden1+Bias1
        Hidden1 = Hidden1.sigm()

        Output = Weight2.dot(Hidden1)
        Output = Output+Bias2
        Output = Output.sigm()

    def cost_function(self, output, expected):
        global StartListBackProp
        StartListBackProp = []
        Sum = 0
        for i in range(len(output)):
            Sum += (output[i][0]-expected[i])**2
            StartListBackProp.append([2 * (output[i][0]-expected[i])])
        return Sum

    def backward_propagation(self):
        global StartBackProp, Hidden1, Weight1, Weight2, Bias1, Bias2, Output

        Bias2Gradient = Output.before_sigm().der_of_sigm().mult_same_size(StartBackProp)
        Bias2 = Bias2 - Bias2Gradient*LearningRate
        Weight2Gradient = Bias2Gradient.dot(Hidden1.transpose())
        Weight2 = Weight2 - Weight2Gradient*LearningRate  # first updated
        StartBackProp = Hidden1 + Weight2Gradient.transpose().sum_right()

        Bias1Gradient = Hidden1.before_sigm().der_of_sigm().mult_same_size(StartBackProp)
        Bias1 = Bias1 - Bias1Gradient*LearningRate
        Weight1Gradient = Bias1Gradient.dot(Input.transpose())
        Weight1 = Weight1 - Weight1Gradient*LearningRate  # second updated
        StartBackProp = Input + Weight1Gradient.transpose().sum_right()
