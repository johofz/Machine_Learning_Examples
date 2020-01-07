from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def best_fit_line(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs*ys)) /
        (mean(xs)**2 - mean(xs**2)))

    b = mean(ys) - m * mean(xs)
    regression_Line = [(m * x + b) for x in xs]
    return m, b, regression_Line
    



if __name__ == '__main__':
    style.use('ggplot')
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,], dtype=np.float64)
    y = np.array([])
    for i in range(len(x)):
        random_num = np.random.rand() * 10.0
        y = np.append(y, random_num)
    y = y.astype(np.float64)

    print(x)
    print(y)

    m, b, regLine = best_fit_line(x, y)
    print(m, b)

    plt.scatter(x, y)
    plt.plot(x, regLine)
    plt.show()
