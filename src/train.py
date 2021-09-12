from droid import Droid
from graphics import *
import numpy as np
import random as rand


# ----constants
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

GEN_AMOUNT = 10
APEX_SURVIVORS = 0.25

MUTATION_RATE = 0.3


# ----setup
def init(amount):
    first_gen = [Droid([2, 5, 5, 2]) for i in range(amount)]
    target_x, target_y = rand.random()*WINDOW_WIDTH, rand.random()*WINDOW_HEIGHT
    target = Point(target_x, target_y)
    return first_gen, target

def main():
    # setup
    win = GraphWin("Simul", WINDOW_WIDTH, WINDOW_HEIGHT)
    win.setBackground(color_rgb(50, 48, 60))

    gen, target_pos = init(GEN_AMOUNT)
    t_greyscale = 165
    t_outline = int(t_greyscale*0.35)

    # draw
    target = Circle(target_pos, 5)
    target.setFill(color_rgb(t_greyscale, t_greyscale, t_greyscale))
    target.setOutline(color_rgb(t_greyscale-t_outline, t_greyscale-t_outline, t_greyscale-t_outline))
    target.draw(win)
    #while True:
    #    pass

    # close
    win.getMouse()
    win.close()


# ----helpers
def perform(it, droids, target):
    # let droids try to seek target using their "brain" (use 2d rendering for python for vis)
    # track score of every droid (score[n] = dist(gen[n], target))
    pos_target = np.array(target.getCenter())
    while it > 0:
        for droid in droids:
            droid.move(pos_target)
        it -= 1

    eval(droids, pos_target)
    new_gen = crossover(droids)
    mutate(new_gen)


def eval(droids, target):
    # use tracked scores and fitness func to select best performing droids
    # with the help of crossover breed new droids
    # slightly mutate new bred droids
    # keep best performing droids from previous gen
    for droid in droids:
        d_t = np.subtract(target, droid.pos)
        dist = np.sum(np.power(d_t, 2))
        droid.score = dist

def crossover(droids, att):     # FIXME: might be wrong/ 'inefficient'
    # randomly select droids from gen to breed new child with their nn-weights/ -biases
    # selection biased by their performance e.g. better score more likely to be selected
    gen_apex = int(len(droids)*APEX_SURVIVORS)
    selected = select_by_fitness(droids, gen_apex)
    new_gen = selected[gen_apex:]

    for i in range(GEN_AMOUNT):
        while att > 0:
            p1 = rand.randint(0, len(selected))
            p2 = rand.randint(0, len(selected))
            if p1 == p2:
                att -= 1
                continue
            new_gen.append(make_child(selected[p1], selected[p2]))
    
    return new_gen

def make_child(p1, p2):
    child = p1
    for i in range(len(p1.brain.network)):
        if i%2 == 0:
            pass        # TODO: select weights from p1 or p2 (50%)
        else:
            continue
    return child

def select_by_fitness(droids, gen_apex):
    sum_scores = 0
    selected = []
    quicksort(droids)

    for droid in droids[gen_apex:]:
        sum_scores += droid.score

    for droid in droids:
        rand = rand.randint(0, sum_scores)
        if gen_apex > 0:
            selected.append(droid)
        else:
            selected.append(droid) if droid.score > rand else selected = selected
        gen_apex -= 1
    return selected

def partition(arr, low, high):
    i = (low-1)
    pivot = arr[high]
  
    for j in range(low, high):
        if arr[j].score <= pivot.score:
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
  
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i+1)
def quicksort(arr, low, high):
    if len(arr) == 1:
        return arr

    if low < high:
        pi = partition(arr, low, high)
  
        quicksort(arr, low, pi-1)
        quicksort(arr, pi+1, high)

def mutate(droids):
    # slightly mutate randomly selected weights
    # whether weight is mutated or not is determined by chance
    for droid in droids:
        for i in range(len(droid.brain.network)):
            if i%2 == 0:    # dense layer ↓
                pass        # TODO: mutate weights slightly
            else:           # activation layer → no weights to be mutated
                continue


# ----run
main()