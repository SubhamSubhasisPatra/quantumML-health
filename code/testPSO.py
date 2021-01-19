import random
import numpy as np
from classical_pso import PSO

w = [5, 5, 4, 4]
bounds=[(-1,1), (-1, 1), (-1, 1), (-1, 1)]
num_particles = 10
max_iter = 1000

def cost_func(x):
    for i in range(0, len(x)):
        return (x[i]/2)

pso_test = PSO(cost_func, w, bounds, num_particles, max_iter)

print(pso_test)