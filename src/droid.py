from brain import Brain
import numpy as np

b = Brain([1, 2, 2])
input = 2
b.decide(input)

class Droid():
    def __init__(self, layer_shapes) -> None:
        self.brain = Brain(layer_shapes)
        self.pos = np.array([0, 0])

    def move(self, target):
        new_pos = self.brain.decide(target)
        self.pos = new_pos

    
