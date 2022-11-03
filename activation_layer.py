from layer import Layer

class activationlayer(Layer):
    def __init__(self, act, act_p):
        self.act = act
        self.act_p = act_p

# This is returning the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.act(self.input)
        return self.output

# This is returning Input error for a given Output error
# Learning Rate is not used because there is no learnable parameters
    def backward_propagation(self, output_err, learning_rate):
        return self.act_p(self.input) * output_err