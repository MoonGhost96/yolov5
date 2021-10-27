import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol,solve


def solve_gb(max_c=768, min_c=32, max_exp=1.6, min_exp=1.1):
    g = Symbol('g')
    b = Symbol('b')
    sol = solve([min_exp * (math.log(max_c, 2) + b) - g, max_exp * (math.log(min_c, 2) + b) - g], [g, b])
    # sol = solve([math.log(max_c,2)-b+min_exp*g, math.log(min_c,2)-b+max_exp*g], [g,b])
    # sol = solve([g*max_exp+b-min_c,g*min_exp+b-max_c],[g,b])
    return sol[g], sol[b]


def get_exp(g, b, c):
    exp = g / (math.log(c, 2) + b)
    # exp = -1*(math.log(c,2)-b)/g
    # exp = (c-b)/g
    return exp


def show_curve(min_c=64, max_c=768):
    g, b = solve_gb(max_c, min_c)
    x = np.arange(min_c, max_c, 1)
    y = []
    for t in x:
        y_1 = get_exp(g, b, t)
        y.append(y_1)
    plt.plot(x, y, label="exp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    show_curve()
