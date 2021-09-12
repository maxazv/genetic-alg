from brain import Brain
from graphics import Point

b = Brain([1, 2, 2])
input = 2
b.decide(input)

class Droid():
    def __init__(self, layer_shapes, pos=Point(0, 0)) -> None:
        self.brain = Brain(layer_shapes)
        self.pos = pos
        self.score = -1

    def move(self, target):
        new_pos = self.brain.decide(target)
        self.pos = Point(new_pos[0][0], new_pos[1][0])

    
