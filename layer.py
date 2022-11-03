class Layer:
    def __init__(self):
        self.input = None
        self.output = None

# Computing output of a layer for a given input
    def forward_propagation(self, input):
        raise NotImplementedError

# Computing dE/dX for a given dE/dY (and updating parameters if any)
    def backward_propagation(self, output_err, learning_rate):
        raise NotImplementedError