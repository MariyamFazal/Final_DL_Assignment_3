class Network:
    def __init__(self):
        self.layers = []
        self.l = None
        self.l_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, l, l_prime):
        self.l = l
        self.l_prime = l_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        S = len(input_data)
        Result = []

        # run network over all samples
        for i in range(S):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            Result.append(output)

        return Result

    # train the network
    def fit(self, x_train, y_train, iterations, learning_rate):
        # sample dimension first
        S = len(x_train)

        # training loop
        for i in range(iterations):
            E = 0
            for j in range(S):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                E += self.l(y_train[j], output)

                # backward propagation
                error = self.l_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            E /= S
            print('epoch %d/%d   error=%f' % (i+1, iterations, E))
