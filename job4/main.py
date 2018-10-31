import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_1 = np.array([0, 1])
    x_2 = np.array([1, 0])

    u = np.random.random([2, 2])
    w = np.random.random([2, 2])

    h = np.zeros([2])
    x = []
    y = []
    for i in range(4):
        h = np.matmul(u, x_1) + np.matmul(w, h)
        print(h)
        x.append(h[0])
        y.append(h[1])
    plt.plot(x, y, "r-")
    h = np.zeros([2])
    x = []
    y = []
    for i in range(4):
        h = np.matmul(u, x_2) + np.matmul(w, h)
        print(h)
        x.append(h[0])
        y.append(h[1])
    plt.plot(x, y, "b-")
    plt.show()
