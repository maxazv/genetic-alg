import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_n, y_n, momentum=False, beta=0.9):
        """Single Neural Network Layer of shape (`output_n`, `data_n`)"""
        self.w = np.random.rand(y_n, x_n) - 0.5
        self.b = np.random.rand(y_n, 1) - 0.5

        self.momentum = momentum
        self.beta = beta
        self.velB = self.velW = 0

    def fforward(self, x):
        self.input = x
        return self.w.dot(x) + self.b
