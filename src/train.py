from droid import Droid

import time
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
    first_gen = [Droid([4, 5, 5, 2], Point(200, 200)) for i in range(amount)]
    target_x, target_y = rand.random()*WINDOW_WIDTH, rand.random()*WINDOW_HEIGHT
    target = Point(target_x, target_y)
    return first_gen, target

def main():
    # -setup-
    win = GraphWin("Simul", WINDOW_WIDTH, WINDOW_HEIGHT, autoflush=False)
    win.setBackground(color_rgb(50, 48, 60))

    iter = 50

    items = []
    gen, target_pos = init(GEN_AMOUNT)
    t_greyscale = 165
    t_outline = int(t_greyscale*0.35)

    for obj in gen:
        graph_obj = Circle(obj.pos, 5)
        graph_obj.setFill(color_rgb(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)))
        graph_obj.setOutline(color_rgb(t_greyscale-t_outline, t_greyscale-t_outline, t_greyscale-t_outline))
        items.append(graph_obj)

    target = Circle(target_pos, 10)
    target.setFill(color_rgb(t_greyscale, t_greyscale, t_greyscale))
    target.setOutline(color_rgb(t_greyscale-t_outline, t_greyscale-t_outline, t_greyscale-t_outline))
    items.append(target)


    # -draw-
    show(items, win)
    for i in range(iter):
        perform(items, gen)
        update(30)
    
    # TODO: test funcs below
    eval(gen, target_pos)
    #new_gen = crossover(gen)
    #mutate(new_gen)


    # -close-
    win.getMouse()
    win.close()


def show(items, win):
    for item in items:
        item.draw(win)



# ----helpers
def perform(items, gen):
    droids, target = items[:GEN_AMOUNT], items[GEN_AMOUNT:][0]
    pos_t = target.getCenter()

    for i in range(len(gen)):
        inp_t = conv_pnt(pos_t)
        inp_d = conv_pnt(gen[i].pos)
        total_inp = np.concatenate((inp_t, inp_d), axis=0)

        dir = gen[i].move(total_inp)
        droids[i].move(dir.getX(), dir.getY())

def conv_pnt(p, shape=(2, 1)):
    return np.array([p.getX(), p.getY()]).reshape(2, 1)


def eval(droids, target):
    # use tracked scores and fitness func to select best performing droids
    # with the help of crossover breed new droids
    # slightly mutate new bred droids
    # keep best performing droids from previous gen
    target = conv_pnt(target)
    for droid in droids:
        conv_pos = conv_pnt(droid.pos)
        d = np.subtract(target, conv_pos)
        dist = np.sum(np.power(d, 2))
        droid.score = 1/dist
        print(droid.score)


def crossover(droids, att):
    # randomly select droids from gen to breed new child with their nn-weights/ -biases
    # selection biased by their performance e.g. better score -> more likely to be selected
    gen_apex = int(len(droids)*APEX_SURVIVORS)
    selected = select_by_fitness(droids, gen_apex)
    new_gen = selected[gen_apex:]

    for i in range(GEN_AMOUNT):
        atmpt = att
        while atmpt > 0:
            p1 = rand.randint(0, len(selected))
            p2 = rand.randint(0, len(selected))
            if p1 == p2:
                atmpt -= 1
                continue
            new_gen.append(make_child(selected[p1], selected[p2]))
    return new_gen

def make_child(p1, p2):     # TODO: test
    child = p1

    for i in range(len(p1.brain.network)):
        if i%2 == 0:
            shape_w, shape_b = p1.brain.network[i].w.shape, p1.brain.network[i].b.size
            p_weights = [p1.brain.network[i].w, p2.brain.network[i].w]
            p_bias = [p1.brain.network[i].b, p2.brain.network[i].b]

            w = np.random.randint(2, size=shape_w) # np-arr 0 or 1 (random)
            cross_w = np.choose(w, p_weights) # every iter. choose weights from p1 or p2 based on w
            
            b = np.random.randint(2, size=shape_b)
            cross_b = np.choose(b, p_bias)
            
            child.brain.network.w, child.brain.network.b = cross_w, cross_b
        else:
            continue
    return child

def select_by_fitness(droids, gen_apex):    # TODO: test
    sum_scores = 0
    selected = []
    quicksort(droids)

    for droid in droids[gen_apex:]:
        sum_scores += droid.score

    for droid in droids:
        if gen_apex > 0:
            selected.append(droid)
            continue

        rand = rand.randint(0, sum_scores)
        if(droid.score > rand):
            selected.append(droid)
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


def mutate(droids):     # TODO: test
    # slightly mutate randomly selected weights
    # whether weight is mutated or not is determined by chance
    for droid in droids:
        for i in range(len(droid.brain.network)):
            if i%2 == 0:    # dense layer
                layer = droid.brain.network[i]

                cond = [layer.w > MUTATION_RATE, layer.w <= MUTATION_RATE]
                choice = [layer.w, layer.w*np.random.normal(-2, 2)]
                mut_weights = np.select(cond, choice)

                droid.brain.network[i] = mut_weights
            else:           # activation layer → no weights, bias to be mutated
                continue



# ----run
main()