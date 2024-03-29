import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_n, y_n):
        self.w = np.random.rand(y_n, x_n) - 0.5
        self.b = np.random.rand(y_n, 1) - 0.5

    def fforward(self, x):
        self.input = x
        return self.w.dot(x) + self.b
