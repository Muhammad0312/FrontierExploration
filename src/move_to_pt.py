#!/usr/bin/python3

import numpy as np
import rospy
import tf
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from std_srvs.srv import Trigger, TriggerRequest, EmptyResponse
import actionlib

from math import sqrt

from utils_lib.online_planning import *
from utils_lib.path_planners import *
# import posePoint.srv
from frontier_explorationb.srv import posePoint
# from 
from frontier_explorationb.msg import go_to_pointAction, go_to_pointGoal, go_to_pointFeedback, go_to_pointResult 

class OnlinePlanner:

    _feedback = go_to_pointFeedback()
    _result = go_to_pointResult()

    # OnlinePlanner Constructor
    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic, dominion, distance_threshold):

        # decide which planner you want
        # Options: RRTStarOMPL, InRRTStar-, FMT-, BIT-, InRRTStar-Dubins, FMT-Dubins, BIT-Dubins, 
        # InRRTStar-BSpline, FMT-BSpline, BIT-BSpline
        self.planner_config = 'RRTStarOMPL'

        # ATTRIBUTE
        # List of points which define the plan. None if there is no plan
        self.path = []
        # State Validity Checker object                                                 
        self.svc = StateValidityChecker(distance_threshold)
        # Current robot SE2 pose [x, y, yaw], None if unknown            
        self.current_pose = None
        # Goal where the robot has to move, None if it is not set                                                                   
        self.goal = None
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        # Dominion [min_x_y, max_x_y] in which the path planner will sample configurations                           
        self.dominion = dominion                                        

        # CONTROLLER PARAMETERS
        # Proportional linear velocity controller gain
        self.Kv = 0.5
        # Proportional angular velocity controller gain                   
        self.Kw = 0.5
        # Maximum linear velocity control action                   
        self.v_max = 0.15
        # Maximum angular velocity control action               
        self.w_max = 0.3      

        # Status
        self.is_moving = False
        self.reached = False         

        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1) 
        # TODO: publisher to cmd_vel_topic
        # Publisher for visualizing the path to with rviz
        self.marker_pub = rospy.Publisher('/turtlebot_online_path_planning/path_marker', Marker, queue_size=1)
        
        # SUBSCRIBERS  ----------------------------------------------------------------

        # TODO: subscriber to gridmap_topic from Octomap Server, cb: get_gridmap
        self.gridmap_sub = rospy.Subscriber("/projected_map", OccupancyGrid, self.get_gridmap) 

        # TODO: subscriber to odom_topic, cb: get_odom 
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.get_odom)

        # rospy.wait_for_service('/set_goal')
        # self.server_set_goal = rospy.ServiceProxy('/set_goal', posePoint)

        self._as = actionlib.SimpleActionServer('move_to_point', go_to_pointAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()

        rospy.Timer(rospy.Duration(0.1), self.controller)
        
    def execute_cb(self, goal):
        # helper variables
        r = rospy.Rate(1)
        success = False
        print('goal received: ',goal.goal_x,goal.goal_y)
        
        self.goal = np.array([goal.goal_x, goal.goal_y])
        self.is_moving = True
        self.path = self.plan()

        self._feedback.dist_to_goal = 0.0
        # publish the feedback
        if self.reached == True:
            print('reached')
            self._result.success = True
            self._as.set_succeeded(self._result)
        
        while not self.reached:
            # print('in while')
            self._feedback.dist_to_goal = self.distance_to_goal()
            self._as.publish_feedback(self._feedback)
            if self.reached:
                # if the goal is reached, set the goal to succeeded
                print('success')
                self._result.success = True
                self._as.set_succeeded(self._result)
                self.reached = False
                break
            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % 'move_to_point')
                self._as.set_preempted()
                success = False
                break

        # print('success')
        # self._result.success = True
        # self._as.set_succeeded(self._result)

    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        # print('odom received')
    
    
        
    # Map callback:  Gets the latest occupancy map published by Octomap server and update 
    # the state validity checker
    def get_gridmap(self, gridmap):
    
        # to avoid map update too often (change value if necessary)
        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:            
            self.last_map_time = gridmap.header.stamp

            # Update State Validity Checker
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(env, gridmap.info.resolution, origin)

            
            # # If robot isnt moving, call frontier to get a new goal
            # if self.is_moving == False:
            #     # Call service to get the initial goal
            #     goal = self.server_set_goal(True)
            #     self.goal = np.array([goal.x, goal.y])
            #     self.is_moving = True
            #     self.path = self.plan() 

            # If the robot is following a path, check if it is still valid
            if self.path is not None and len(self.path) > 0:
                # create total_path adding the current position to the rest of waypoints in the path
                total_path = [self.current_pose[0:2]] + self.path
                if self.svc.check_path(total_path) == False:
                    print("Invaluid Path")
                    self.path = self.plan() 



    # Solve plan from current position to self.goal. 
    def plan(self):
        # List of waypoints [x, y] that compose the plan
        path = []
        trial = 0
        # If planning fials, allow replanning for several trials
        while len(path) <= 1 and trial < 5:
            print("Compute new path")

            # TODO: plan a path from self.current_pose to self.goal
            path = compute_path(self.planner_config, self.current_pose, self.goal,self.svc, self.dominion, max_time=3.0) 
            trial += 1

        if trial == 5:
            # If planning fails, consider increasing the planning time
            print("Path not found! Let's try a nearby point")

            # This condition ensure that if we land on obstacle map, we chose the a nearby free space as starting point of planning doesnt fail
            start_pt = self.current_pose
            while len(path) == 0:
                start_pt[0] += 0.1
                start_pt[1] += 0.1
                try:
                    path = compute_path(self.planner_config, start_pt, self.goal,self.svc, self.dominion, max_time=3.0)
                except:
                    pass
            print("Path found")
            # Publish plan marker to visualize in rviz
            # print(path)
            self.publish_path(path)
            # Do NOT delete initial point, as initial point is not current pose.
        else:
            print("Path found")
            # Publish plan marker to visualize in rviz
            # print(path)
            self.publish_path(path)
            # remove initial waypoint in the path (current pose is already reached)
            del path[0]                 
        return path


    # This method is called every 0.1s. It computes the velocity comands in order to reach the 
    # next waypoint in the path. It also sends zero velocity commands if there is no active path.
    def controller(self, event):
        v = 0
        w = 0
        if self.path is not None and len(self.path) > 0:
            
            # If current waypoint reached with some tolerance move to next way point, otherwise move to current point
            if np.linalg.norm(self.path[0] - self.current_pose[0:2]) < 0.1:
                # print (" self.current_pose[0:2]", self.current_pose[0:2],"self.svc.resolution",self.svc.resolution)
                # print("---------------------------------------------")
                # print("path point",self.path[0],"current_pose",self.current_pose)
                # print("Position {} reached".format(self.path[0]))
                del self.path[0]
                if len(self.path) == 0:
                    self.goal = None
                    print("Final position reached!")
                    self.is_moving = False
                    self.reached = True
                    # self._result.success = True
                    # self._as.set_succeeded(self._result)
            else:
                # TODO: Compute velocities using controller function in utils_lib
                v,w = move_to_point(self.current_pose, self.path[0], self.Kv, self.Kw)
                self.is_moving = True
                # print(v,w)
        
        # Publish velocity commands
        self.__send_command__(v, w)
    

    # PUBLISHER HELPERS

    # Transform linear and angular velocity (v, w) into a Twist message and publish it
    def __send_command__(self, v, w):
        cmd = Twist()
        cmd.linear.x = np.clip(v, -self.v_max, self.v_max)
        cmd.linear.y = 0
        cmd.linear.z = 0
        cmd.angular.x = 0
        cmd.angular.y = 0
        cmd.angular.z = np.clip(w, -self.w_max, self.w_max)
        self.cmd_pub.publish(cmd)

    # Publish a path as a series of line markers
    def publish_path(self, path):
        if len(path) > 1:
            print("Publish path!")
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = rospy.Time.now()
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 1
            color_red.g = 0
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_red)
            
            self.marker_pub.publish(m)
        
    def distance_to_goal(self):        #distance between the current pose and the goal pose 
        if self.goal is None:
            return 0.0
        return sqrt(( self.goal[0] - self.current_pose[0])**2 + ( self.goal[1] - self.current_pose[1])**2)
        
# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('turtlebot_online_path_planning_node')   
    ##############     neigborhood size comparison    #######################
    node = OnlinePlanner('/projected_map', '/odom', '/cmd_vel', np.array([-15.0, 15.0]), 0.3)
    
    # Run forever
    rospy.spin()
