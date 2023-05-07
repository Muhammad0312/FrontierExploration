import math
import numpy as np
from scipy import interpolate
from collections import OrderedDict

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

def move_to_point(current, goal, Kp=10, Ki=10, Kd=10, dt=0.05):
    # Compute distance and angle to goal
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    dist = math.sqrt(dx**2 + dy**2)
    angle = wrap_angle(math.atan2(dy, dx) - current[2])

    # Compute errors
    error_dist = dist
    error_angle = angle

    # Store previous errors
    if 'prev_error_dist' not in move_to_point.__dict__:
        move_to_point.prev_error_dist = error_dist
    if 'prev_error_angle' not in move_to_point.__dict__:
        move_to_point.prev_error_angle = error_angle

    # Compute PID terms
    error_dist_deriv = (error_dist - move_to_point.prev_error_dist)
    error_angle_deriv = (error_angle - move_to_point.prev_error_angle)
    error_dist_integral = (error_dist + move_to_point.prev_error_dist)
    error_angle_integral = (error_angle + move_to_point.prev_error_angle)

    v = Kp * error_dist + Ki * error_dist_integral + Kd * error_dist_deriv
    w = Kp * error_angle + Ki * error_angle_integral + Kd * error_angle_deriv

    # Update previous errors
    move_to_point.prev_error_dist = error_dist
    move_to_point.prev_error_angle = error_angle

    # # Limit angular velocity to avoid overshooting
    if abs(angle) > 0.2:
        v = 0

    return v, w


def distance_between_points(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def points_on_line(self, x1, y1, x2, y2, n):
    dx = (x2 - x1) / (n - 1)
    dy = (y2 - y1) / (n - 1)
    return [(x1 + i * dx, y1 + i * dy) for i in range(n - 1)] + [(x2, y2)]

def smooth_path_bspline(self, waypoints):
    interpolated_points = []

    for i in range(0, len(waypoints) - 1):
        num_of_points = int(self.distance_between_points(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1]))
        new_points = self.points_on_line(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1], (num_of_points)+2)
        for p in new_points:
            interpolated_points.append(p)

    unique_dict = OrderedDict.fromkeys(interpolated_points)
    unique_list = list(unique_dict.keys())

    x = []
    y = []

    for point in unique_list:
        x.append(point[0])
        y.append(point[1])
    
    tck, *rest = interpolate.splprep([x, y], s=0.001)
    u = np.linspace(0, 1, num=100)
    smooth= interpolate.splev(u, tck)

    path = []
    for i in range(0, len(smooth[0])):
        path.append((smooth[0][i], smooth[1][i]))


    return path