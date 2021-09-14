from brain import Brain
from graphics import Point

SCALAR = 3

class Droid():
    def __init__(self, layer_shapes, pos=Point(0, 0)) -> None:
        self.brain = Brain(layer_shapes)
        self.pos = pos
        self.score = -1
        self.norm = 0

        self.alive = True

    def reset(self, pos):
        self.pos = pos
        self.score = -1
        self.norm = 0

    def move(self, target):
        new_pos = self.brain.decide(target)

        dir = Point(new_pos[0][0]*SCALAR, new_pos[1][0]*SCALAR)
        self.pos.x += dir.x
        self.pos.y += dir.y

        return dir

    
