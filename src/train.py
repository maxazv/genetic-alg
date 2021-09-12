# TODO: init multiple droids with random 'brains'
#       make fitness function (e.g. selection)
#       add mutations, crossover
#       further research, evaluate input/ output nodes of brain
from droid import Droid
import numpy as np
import random as rand


WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
GEN_AMOUNT = 10

def init(amount):
    first_gen = [Droid([2, 5, 5, 2]) for i in range(amount)]
    target_x, target_y = rand.random()*WINDOW_WIDTH, rand.random()*WINDOW_HEIGHT
    target = np.array([target_x, target_y])
    return first_gen, target


gen, target = init(GEN_AMOUNT)

def perform():
    # let droids try to seek target using their "brain" (use 2d rendering for python for vis)
    # track score of every droid
    pass

def eval():
    # use tracked scores and fitness func to select best performing droids
    # with the help of crossover breed new droids
    # slightly mutate new bred droids
    # keep best performing droids from previous gen
    pass

def crossover():
    # randomly select droids from gen to breed new child with their nn-weights/ -biases
    # selection biased by their performance e.g. better score more likely to be selected
    pass

def mutate():
    # slightly mutate randomly selected weights
    # whether weight is mutated or not is determined by chance
    pass
