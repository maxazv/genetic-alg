from hashlib import new
from tkinter import Canvas
from typing import Dict
from droid import Droid

from graphics import *
import numpy as np
import random as rand
import math



# ----constants
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

GEN_AMOUNT = 50
APEX_SURVIVORS = 0.25

MUTATION_RATE = 0.3
MUT_AMOUNT = 0.7

verbose = True



# ----setup
def init(amount):
    first_gen = [Droid([4, 5, 5, 2], Point(200, 200)) for i in range(amount)]
    target_x, target_y = rand.random()*WINDOW_WIDTH, rand.random()*WINDOW_HEIGHT
    target = Point(target_x, target_y)
    return first_gen, target

def main(verbose):
    # -setup-
    win = GraphWin("Simul", WINDOW_WIDTH, WINDOW_HEIGHT, autoflush=False)
    win.setBackground(color_rgb(50, 48, 60))

    steps = 15
    iter = 20

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
    show(items[GEN_AMOUNT:], win)
    for i in range(iter):
        show(items[:GEN_AMOUNT], win)
        for j in range(steps):
            perform(items, gen)
            update(30)

        eval(gen, target_pos)
        gen = gen[::-1]
        new_gen = crossover(gen, 10)
        mutate(new_gen)
        print_score(gen, 0, int(GEN_AMOUNT*0.3), "Apex Droids:")
        print(len(new_gen))
        gen = new_gen
        items = items[GEN_AMOUNT:]
        items[0:0] = [Circle(x.pos, 5) for x in gen]

        update()

    if verbose:
        for d in gen:
            line(target_pos, d.pos, win)

            t = Text(Point(d.pos.x, d.pos.y-13), str(d.score)).draw(win)
            t.setFill(color_rgb(115, 112, 119))
            t.setSize(10)

    # -close-
    win.getMouse()
    win.close()


def show(items, win):
    for item in items:
        item.draw(win)
def line(p1, p2, win, color=color_rgb(115, 112, 119)):
    l = Line(p1, p2)
    l.setFill(color)
    l.setWidth(0.05)
    l.draw(win)
def print_score(gen, start, end, msg=""):
    print(msg, [x.score for x in gen[start:end-1]])


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
def dist(p1, p2):
    dx, dy = p1.getX()-p2.getX(), p2.getY()-p1.getY()
    dist = math.sqrt(sum([pow(dx, 2), pow(dy, 2)]))
    return dist


def eval(droids, target):
    # use tracked scores and fitness func to select best performing droids
    # with the help of crossover breed new droids
    # slightly mutate new bred droids
    # keep best performing droids from previous gen
    for droid in droids:
        if droid.pos.x < 0 or droid.pos.y < 0 or droid.pos.x > WINDOW_WIDTH and droid.pos.y > WINDOW_HEIGHT:
            droid.alive = False
        dx = target.x - droid.pos.x
        dy = target.y - droid.pos.y
        dist = math.sqrt(sum([pow(dx, 2), pow(dy, 2)]))
        droid.score = int((10/(dist-(dist*0.75)))*100)

    c=0
    for i in range(len(droids)-c):
        if not droids[i-c].alive:
            droids.pop(i-c)
            c+=1

    quicksort(droids, 0, len(droids)-1)
    droids = droids[::-1]



# TODO: test
def crossover(droids, att):
    # randomly select droids from gen to breed new child with their nn-weights/ -biases
    # selection biased by their performance e.g. better score -> more likely to be selected
    gen_apex = int(len(droids)*APEX_SURVIVORS)
    selected = select_by_fitness(droids, gen_apex)

    new_gen = selected[:gen_apex]
    for i in range(GEN_AMOUNT-gen_apex):
        atmpt = att
        while atmpt > 0:
            # FIXME
            p1 = rand.randint(0, len(selected)-1)
            p2 = rand.randint(0, len(selected)-1)
            if p1 == p2:
                atmpt -= 1
                continue

            new_gen.append(make_child(selected[p1], selected[p2]))
            break
    return new_gen

def make_child(p1, p2):
    child = p1
    child.origin, child.pos = p1.origin, p1.origin
    for i in range(len(p1.brain.network)):
        if i%2 == 0:
            shape_w, shape_b = p1.brain.network[i].w.shape, p1.brain.network[i].b.size
            p_weights = [p1.brain.network[i].w, p2.brain.network[i].w]
            p_bias = [p1.brain.network[i].b, p2.brain.network[i].b]

            w = np.random.randint(2, size=shape_w) # np-arr 0 or 1 (random)
            cross_w = np.choose(w, p_weights) # every iter. choose weights from p1 or p2 based on w
            
            b = np.random.randint(2, size=(shape_b, 1))
            cross_b = np.choose(b, p_bias)
            
            child.brain.network[i].w, child.brain.network[i].b = cross_w, cross_b
        else:
            continue
    return child

def select_by_fitness(droids, gen_apex):
    sum_scores = 0
    selected = []
    #quicksort(droids, 0, len(droids)-1)
    #droids = droids[::-1]   # reversing list

    for droid in droids[gen_apex:]:
        sum_scores += droid.score

    for droid in droids:
        if gen_apex > 0:
            selected.append(droid)
            continue
        
        prob = rand.randint(0, sum_scores)
        if(droid.score > prob):
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


def mutate(droids):
    # slightly mutate randomly selected weights
    # whether weight is mutated or not is determined by chance
    for droid in droids:
        for i in range(len(droid.brain.network)):
            if i%2 == 0:    # dense layer
                layer = droid.brain.network[i]
                prob_w = np.random.uniform(0, 1, size=layer.w.shape)
                prob_b = np.random.uniform(0, 1, size=(layer.b.size, 1))

                cond_w = [prob_w > MUTATION_RATE, prob_w <= MUTATION_RATE]
                cond_b = [prob_b > MUTATION_RATE, prob_b <= MUTATION_RATE]

                choice_w = [layer.w, layer.w*(np.random.normal(-2, 2)*MUT_AMOUNT)]
                choice_b = [layer.b, layer.b*(np.random.normal(-2, 2)*MUT_AMOUNT)]

                mut_weights = np.select(cond_w, choice_w)
                mut_bias = np.select(cond_b, choice_b)

                droid.brain.network[i].w = mut_weights
                droid.brain.network[i].b = mut_bias
            else:           # activation layer → no weights, bias to be mutated
                continue



# ----run
main(verbose)