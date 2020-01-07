from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

def create_dataset(entries, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(entries):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64) 


def best_fit_line(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs*ys)) /
        (mean(xs)**2 - mean(xs**2)))

    b = mean(ys) - m * mean(xs)
    regression_Line = [(m * x + b) for x in xs]
    return m, b, regression_Line


def squared_error(ys_original, ys_line):
    return sum((ys_line-ys_original)**2)


def coeff_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_reg = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return (1 - squared_error_reg / squared_error_y_mean)


if __name__ == '__main__':
    style.use('ggplot')
    
    x, y = create_dataset(20, 10, 2, correlation='pos')

    print(x)
    print(y)

    m, b, regLine = best_fit_line(x, y)
    coeff = coeff_determination(y, regLine)
    print(m, b, coeff)

    plt.scatter(x, y)
    plt.plot(x, regLine)
    plt.show()
