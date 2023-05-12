import math
import numpy as np
from scipy import interpolate
from collections import OrderedDict

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

def move_to_point_smooth(current, goal, Kp=10, Ki=10, Kd=10, dt=0.05):
    # Compute distance and angle to goal
    # print('Goal: ', goal)
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    dist = math.sqrt(dx**2 + dy**2)
    angle = wrap_angle(math.atan2(dy, dx) - current[2])

    # Compute errors
    error_dist = dist
    error_angle = angle

    # Store previous errors
    if 'prev_error_dist' not in move_to_point_smooth.__dict__:
        move_to_point_smooth.prev_error_dist = error_dist
    if 'prev_error_angle' not in move_to_point_smooth.__dict__:
        move_to_point_smooth.prev_error_angle = error_angle

    # Compute PID terms
    error_dist_deriv = (error_dist - move_to_point_smooth.prev_error_dist)
    error_angle_deriv = (error_angle - move_to_point_smooth.prev_error_angle)
    error_dist_integral = (error_dist + move_to_point_smooth.prev_error_dist)
    error_angle_integral = (error_angle + move_to_point_smooth.prev_error_angle)

    v = Kp * error_dist + Ki * error_dist_integral + Kd * error_dist_deriv
    w = Kp * error_angle + Ki * error_angle_integral + Kd * error_angle_deriv

    # Update previous errors
    move_to_point_smooth.prev_error_dist = error_dist
    move_to_point_smooth.prev_error_angle = error_angle

    # # Limit angular velocity to avoid overshooting
    if abs(angle) > 0.2:
        v = 0
    print(v,w)
    return v, w