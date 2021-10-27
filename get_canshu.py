import math
import matplotlib.pyplot as plt
import numpy as np

def get(c):
    gamma = 12.96
    b = 1.2
    exp = gamma / (math.log(c, 2) + b)
    exp = round(exp / 0.01) * 0.01
    return exp

def show_quxian():
    x = np.arange(64, 1024, 1)
    y = []
    for t in x:
        y_1 = get(t)
        y.append(y_1)
    plt.plot(x, y, label="exp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

print(get(1024))
