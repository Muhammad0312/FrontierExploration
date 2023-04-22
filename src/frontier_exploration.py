#!/usr/bin/python3
from asyncore import loop
import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
import roslib
import tf
import cv2
import os
from std_srvs.srv import Trigger, TriggerRequest
from pick_up_objects_task.srv import posePoint
from utils_lib.frontier_classes import FrontierDetector

from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class FrontierExplorer:       
    def __init__(self):
    
        # Current robot pose [x, y, yaw], None if unknown            
        self.current_pose = None
        
        self.motion_busy = None

        self.frontierDetector = FrontierDetector()
            
        # Subscribe to map: Every time new map appears, recompute frontiers 
        self.map_subscriber = rospy.Subscriber("/projected_map", OccupancyGrid, self.projected_map_callback)

        # Subscribe to robot pose: Get robot pose
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.get_odom)

        # Publish: Frontier Points for Visualization
        self.frontier_points_pub = rospy.Publisher("/frontier_detection/vis_points",MarkerArray,queue_size=1) #occupancy grid publisher

        self.marker_Arr = MarkerArray()
        self.marker_Arr.markers = []

        

        # service used to set goal of move behaviour
        rospy.wait_for_service('/set_goal')
        # service used to check status of move behaviour
        rospy.wait_for_service('/check_reached')


        self.server_set_goal = rospy.ServiceProxy(
                '/set_goal', posePoint)
        self.server_check_reached = rospy.ServiceProxy(
                '/check_reached', Trigger)
        

    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])

    def projected_map_callback(self, data):
        '''
        Called when a new occupancy grid is received. 
        Does ALL THE WORK FFS
        ''' 

        # If the robot is currently not moving, give it the highest priority candidate point (closes)
        resp = self.server_check_reached(TriggerRequest())
        print(resp.success)
        if resp.success:     
            self.frontierDetector.set_mapNpose(data, self.current_pose)

            candidate_pts_ordered = self.frontierDetector.getCandidatePoint(criterion='entropy')

            candidate_pts_catesian = self.frontierDetector.all_map_to_position(candidate_pts_ordered)
        
            print('selected candidate: ',candidate_pts_ordered[0,0], candidate_pts_ordered[0,1])

            # Publishing
            # self.publish_frontier_points(candidate_pts_catesian)

            # print(candidate_pts_catesian[0,0], candidate_pts_catesian[0,1])
            self.publish_frontier_points([[candidate_pts_catesian[0,0], candidate_pts_catesian[0,1]]])

            self.server_set_goal(candidate_pts_catesian[0,0],candidate_pts_catesian[0,1])


    def flatten_array(self,lst):
        flat_lst = []
        for obj in lst:
            for item in obj:
                flat_lst.append(item)
        return flat_lst
    
    # Publish a path as a series of line markers
    def publish_frontier_points(self,data):   
        self.marker_Arr.markers = []
        for i in range(0,len(data)):
            self.myMarker = Marker()
            self.myMarker.header.frame_id = "odom"
            self.myMarker.type = self.myMarker.SPHERE # sphere
            self.myMarker.action = self.myMarker.ADD

            self.myPoint = Point()
            self.myPoint.x = data[i][0]
            self.myPoint.y = data[i][1]
            self.myMarker.pose.position = self.myPoint
            
            self.myMarker.color=ColorRGBA(0, 1, 0, 1)
            self.myMarker.scale.x = 0.2
            self.myMarker.scale.y = 0.2
            self.myMarker.scale.z = 0.2
            self.myMarker.lifetime = rospy.Duration(0)
            
            self.marker_Arr.markers.append(self.myMarker)
            id = 0
            for m in self.marker_Arr.markers:
                m.id = id
                id += 1
            self.frontier_points_pub.publish(self.marker_Arr)
    

    '''
    Convert map position to world coordinates. 
    '''
    def __map_to_position__(self, p):
        mx = p[1]*self.map_resolution+self.map_origin[0] 
        my = p[0]*self.map_resolution+self.map_origin[1] 
        return [mx,my]


    '''
    Converts a list of points in map coordinates to world coordinates
    '''
    def __all_map_to_position__(self, pts):
        lst = []
        for p in pts:
            lst.append(self.__map_to_position__(p))
        return np.array(lst)

if __name__ == '__main__':
    rospy.init_node('frontier_explorer', anonymous=True)
    n = FrontierExplorer()
    rospy.spin()
