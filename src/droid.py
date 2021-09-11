from brain import Brain
import numpy as np

b = Brain([1, 2, 2])
input = 2
b.decide(input)

class Droid():
    def __init__(self) -> None:
        self.brain = Brain([2, 5, 5, 2])
        self.pos = np.array([])

    def move(self, target):
        new_pos = self.brain.decide(target)
        self.pos = new_pos

    
