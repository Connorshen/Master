import numpy as np


def get_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


if __name__ == '__main__':
    x_dim = 10
    w_dim = 5

    x_1_0 = np.array([0 if i < 5 else 1 for i in range(x_dim)])
    print("x_1_0 =", x_1_0)
    x_1_1 = x_1_0
    x_2_0 = np.array([0 if i >= 5 else 1 for i in range(x_dim)])
    print("x_2_0 =", x_2_0)
    x_2_1 = x_2_0

    w_in = np.random.random((w_dim, x_dim))
    w_r = np.random.random((w_dim, w_dim))
    print("w_in = ", w_in)
    print("w_r =", w_r)

    y_1_1 = np.matmul(w_in, x_1_0)
    y_1_2 = np.matmul(w_in, x_1_1) + np.matmul(w_r, y_1_1)
    y_2_1 = np.matmul(w_in, x_2_0)
    y_2_2 = np.matmul(w_in, x_2_1) + np.matmul(w_r, y_2_1)
    print("y_1_1 = ", y_1_1)
    print("y_1_2 = ", y_1_2)
    print("y_2_1 = ", y_2_1)
    print("y_2_2 = ", y_2_2)
    print("y_1_1 - y_1_2 =", y_1_1 - y_1_2)
    print(np.matmul(np.matmul(w_r, w_in), x_1_0))
    print("y_2_1 - y_2_2 =", y_2_1 - y_2_2)

    print(get_angle(y_1_1 - y_1_2, y_2_1 - y_2_2))
