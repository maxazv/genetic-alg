from dense import Dense
from activations import Tanh

class Brain():
    def __init__(self, layer_shapes) -> None:
        self.layer_shapes = layer_shapes
        self.network = []
        self.setup()


    def setup(self):
        for i in range(len(self.layer_shapes)-1):
            self.network.append(Dense(self.layer_shapes[i], self.layer_shapes[i+1]))
            self.network.append(Tanh)


    def decide(self, x):
        output = x
        for layer in self.network:
            output = layer.fforward(output)
        return output
