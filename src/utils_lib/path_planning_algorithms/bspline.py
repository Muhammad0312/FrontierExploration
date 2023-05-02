from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import math

def distance_between_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def points_on_line(x1, y1, x2, y2, n):
    dx = (x2 - x1) / (n - 1)
    dy = (y2 - y1) / (n - 1)
    return [(x1 + i * dx, y1 + i * dy) for i in range(n - 1)] + [(x2, y2)]


def bspline_smooth(path):
    sorted_path = sorted(path, key=lambda x: x[0])
    interpolated_points = []

    for i in range(0, len(sorted_path) - 1):
        num_of_points = int(distance_between_points(sorted_path[i][0], sorted_path[i][1], sorted_path[i+1][0], sorted_path[i+1][1]))
        new_points = points_on_line(sorted_path[i][0], sorted_path[i][1], sorted_path[i+1][0], sorted_path[i+1][1], num_of_points)
        for p in new_points:
            interpolated_points.append(p)
    sorted_path = list(set(interpolated_points))
    sorted_path = sorted(sorted_path, key=lambda x: x[0])

    x = []
    y = []
    print(num_of_points)
    for vertex in sorted_path:
        x.append(vertex[0])
        y.append(vertex[1])

    tck = interpolate.splrep(x, y, s=0, k=2)
    x_new = np.linspace(min(x), max(x), 100)
    y_fit = interpolate.BSpline(*tck)(x_new)

    smoothed_path = []
    for i in range(len(x_new)):
        smoothed_path.append([x_new[i], y_fit[i]])


    # plt.plot(x, y, 'bo', label='Control Points')
    # plt.plot(x_new, y_fit, 'r-', label='Smoothed Path')
    # plt.legend()
    # plt.show()

    return smoothed_path