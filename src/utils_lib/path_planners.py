import numpy as np
import math
from ompl import base as ob
from ompl import geometric as og

from utils_lib.path_planning_algorithms.fmt import *
from utils_lib.path_planning_algorithms.batch_informed_trees import BITStar
from utils_lib.path_planning_algorithms.dubins_path import dupin_smooth
from utils_lib.path_planning_algorithms.bspline import *
from utils_lib.path_planning_algorithms.clean_informed import IRrtStar

# function that choses the planner based on the inputs
def compute_path(planner_config,start_p, goal_p, state_validity_checker, dominion, max_time=2.0):    
    if planner_config == 'RRTStarOMPL':
        print(planner_config)
        return RRTStarOMPL(start_p, goal_p, state_validity_checker.is_valid, dominion, max_time)
    else:
        planner, smoothing = planner_config.split('-') 
        print(planner, smoothing)
        # return []
        if planner == 'InRRTStar':
            step_len= 1
            goal_sample_rate=0.1
            search_radius=20
            iter_max=2000
            rrt_star = IRrtStar(start_p[0:2], goal_p, step_len, 
                                goal_sample_rate, search_radius, 
                                iter_max, dominion, state_validity_checker)
            path_x,path_y= rrt_star.planning()
            path_x, path_y = path_x[::-1], path_y[::-1]
            path = x_y_to_xy(path_x,path_y)
            # print(path)
            if smoothing == 'Dubins':
                max_c = 5.0
                path = Add_theta(path_x,path_y)
                path=dupin_smooth(path, max_c)
                return path
            elif smoothing == 'BSpline':
                if len(path) > 2:
                    path = bspline_smooth(path)
                return path
            else:
                return x_y_to_xy(path_x,path_y)


        elif planner == 'FMT':
            sample_numbers = 200
            search_radius = 600
            fmt = FMT(start_p[0:2], goal_p, search_radius,dominion,sample_numbers,state_validity_checker)
            path_x,path_y=fmt.Planning()
            path_x, path_y = path_x[::-1], path_y[::-1]
            path = x_y_to_xy(path_x,path_y)
            # print(path)
            if smoothing == 'Dubins':
                max_c = 5.0
                path = Add_theta(path_x,path_y)
                path=dupin_smooth(path, max_c)
                return path
            elif smoothing == 'BSpline':
                if len(path) > 2:
                    path = bspline_smooth(path)
                return path
            else:
                return x_y_to_xy(path_x,path_y)
            
        elif planner == 'BIT':
            eta = 40
            iter_max = 1300
            bit = BITStar(start_p[0:2], goal_p, eta, iter_max,dominion,state_validity_checker)
            path_x,path_y = bit.planning()
            
            path_x, path_y = path_x[::-1], path_y[::-1]
            path = x_y_to_xy(path_x,path_y)
            
            # print(path)
            if smoothing == 'Dubins':
                max_c = 10.0
                path = Add_theta(path_x,path_y,start_p[2])
                path=dupin_smooth(path, max_c)
                return path
            elif smoothing == 'BSpline':
                if len(path) > 2:
                    path = bspline_smooth(path)
                return path
            else:
                return x_y_to_xy(path_x,path_y)            

def x_y_to_xy(path_x,path_y):
    path = []
    for i in range(len(path_x)):
        path.append((path_x[i], path_y[i]))
    return path

def RRTStarOMPL(start_p, goal_p, state_validity_checker, dominion, max_time=2.0):
    space = ob.RealVectorStateSpace(2)
    bound_low=    dominion[0]
    bound_high= dominion[1]
    space.setBounds(bound_low, bound_high)
    space.setLongestValidSegmentFraction(0.001)
    si = ob.SpaceInformation(space) 
    si.setStateValidityChecker(ob.StateValidityCheckerFn(state_validity_checker))
        # create a start state
    start = ob.State(space)
    start[0] = start_p[0]
    start[1] = start_p[1]
    
    # create a goal state
    goal = ob.State(space)
    goal[0] = goal_p[0]
    goal[1] = goal_p[1]

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    optimizingPlanner = og.RRTstar(si)
    optimizingPlanner.setRange(10)  
    optimizingPlanner.setGoalBias(0.2)

    # Set the problem instance for our planner to solve and call setup
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(max_time)
    
    # Get planner data
    pd = ob.PlannerData(si)
    optimizingPlanner.getPlannerData(pd)
    
    if solved:
        # get the path and transform it to a list
        path = pdef.getSolutionPath()
        print("Found solution:\n%s" % path)
        ret = []
        for i in path.getStates():
            ret.append((i[0], i[1]))
    else:
        ret = []
        print("No solution found")
    # print ("path", path)
    return ret
        
def Add_theta(path_x,path_y,curr_theta):
    path = []
    for i in range(len(path_x)-1):
        x0, y0 = path_x[i], path_y[i]
        x1, y1 = path_x[i+1], path_y[i+1]
        dx, dy = x1 - x0, y1 - y0
        angle = math.atan2(dy, dx) * 180 / math.pi
        angle = round(angle / 30) * 30
        path.append((x0, y0, angle))
    path.append((path_x[-1], path_y[-1], angle))
    # print('curr theta: ', curr_theta)
    path[0] = (path[0][0], path[0][1], 0)
    # print(path[0])
    return path