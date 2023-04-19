#!/usr/bin/python3
from asyncore import loop
import rospy
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

from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# import 
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cv2
import matplotlib.pyplot as plt
import time
import skimage
import math
from skimage import measure, draw

class Frontier_explorer:       
    def __init__(self):
        self.entropy = 0 
        self.r = 0 #width of the map
        self.c = 0 #height of the map

        self.map_origin = [0.0,0.0]
        self.map_resolution = 0.0

        # Current robot pose [x, y, yaw], None if unknown            
        self.current_pose = None
        
        self.motion_busy = None
            
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

        # Fetch meta deta of map: 
        map_r1 = data.info.height  #y
        map_c1 = data.info.width   #x
        self.map_origin = [data.info.origin.position.x, data.info.origin.position.y] 
        self.map_resolution = data.info.resolution
        
        # stiore occupancu grid map: 
        occupancy_map = np.array(data.data).reshape(map_r1,map_c1)
        (self.r, self.c) = occupancy_map.shape 
	
        #identify frontiers: Binarization
        frontiers = self.identify_frontiers(occupancy_map)        

	    # label the diff frontiers: Segmentation
        candidate_pts = self.label_frontiers(frontiers,occupancy_map)         
    
	    # Select Points: Candidate Point Selection
        candidate_pts_ordered = self.select_point(candidate_pts, occupancy_map)
    
        print('selected candidate: ',candidate_pts_ordered[0,0], candidate_pts_ordered[0,1])

        # Publishing
        candidate_pts_catesian = self.__all_map_to_position__(candidate_pts_ordered)
        # self.publish_frontier_points(candidate_pts_catesian)

        # print(candidate_pts_catesian[0,0], candidate_pts_catesian[0,1])
        self.publish_frontier_points([[candidate_pts_catesian[0,0], candidate_pts_catesian[0,1]]])


        # If the robot is currently not moving, give it the highest priority candidate point (closes)
        resp = self.server_check_reached(TriggerRequest())
        print(resp.success)
        if resp.success:
            self.server_set_goal(candidate_pts_catesian[0,0],candidate_pts_catesian[0,1])

        # candidate_pts_ordered_send = Int32MultiArray()
        # candidate_pts_ordered_send.data = candidate_pts_ordered
        # #print('candidate pts: ', candidate_pts_ordered_send)
        # self.frontier_grid_goal_list.publish(candidate_pts_ordered_send)


    def identify_frontiers(self, octomap):
        '''
        Identifies frontiers i.e. region between free space and unknown space. 
        Input: 
        octomap -> occupancy grid [-1, 0, 100]
        
        Output:
        result -> Filtered occupancy grid with frontier cells with value 255 (any max value), others 0
        See media -> frontiersX.png for reference 
        '''

        # binary image: 255 for frontier cells, otherwise
        result = np.zeros(shape = (self.r, self.c))

        # iterate over cells: avoid boundary cells 
        for i in range(7, self.r-7):
            for j in range(7, self.c-7):  
                # if cell is free, any neighborhood is unknown, it is a frontier
                flag = False  # flag to break out of both for loops if unknown cell detected
                if octomap[i, j] == 0:
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            if octomap[i+a, j+b] == -1:
                                result[i, j] = 255
                                flag = True
                                break
                        if flag == True:
                            break
                
        demo_map = copy.deepcopy(result)
        cv2.imwrite('frontiers.png',demo_map)       
        return result

    def label_frontiers(self,frontiers,occupancy):
        '''
        1) Segments individual frontiers (clustering)
        2) Filters noise frontiers
        3) Selects centroid of each frontier as the candiate points

        Inputs:
        frontiers -> Binary Grid: Frontier cells = 255, otherwise zero
        occupancy -> Occupancy Grid

        Outputs:
        pts -> candiate (goal) points. One point (centroid) per frontier

        '''

        # Output list of candidate points
        candidate_pts = []

        # Segment/cluster/label the frontiers
        # Background has 0 value, and each frontier has a unique ID (1,2,3)
        labelled_frontiers = measure.label(frontiers, background=0)
        img_norm = cv2.normalize(labelled_frontiers, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the image to uint8 type
        label_img = img_norm.astype(np.uint8)
        # Apply a color map to the image
        color_img = cv2.applyColorMap(label_img, cv2.COLORMAP_HOT)
        cv2.imwrite("clustered.png",color_img)
        
        # Call regionprops function to get characteristics of each frontier
        # Example: size, orientation, centroid
        regions = measure.regionprops(labelled_frontiers)
        
        #__________________________  IMAGE SAVING  _________________________________
        grid = copy.deepcopy(occupancy)
        occu = copy.deepcopy(grid)
        grid[np.where(grid==-1)]=127 # Unknown space
        grid[np.where(grid==0)]=255 # free space
        grid[np.where(grid==100)]=0 # occupied space
        cv2.imwrite("testing.png",grid)
        grid = cv2.imread("testing.png")
        #_________________________________________________________________________________


        for prop in regions:
            # avoid small frontiers (caused by map noise)
            if prop.area > 9.0:
                # get centroid of each frontier using regionprops property
                x = int(prop.centroid[0])
                y = int(prop.centroid[1])

                #__________________________ IMAGE SAVING  _________________________________
                grid = cv2.circle(grid, (int(y),int(x)), 1, (0,0,255),2)
                cv2.imwrite("frontiercenter.png",grid)
                #_________________________________________________________________________________
                
                # Save all centroid in a list that serves as candidate points
                potential_point = [int(prop.centroid[0]),int(prop.centroid[1])] 
                candidate_pts.append([potential_point[0],potential_point[1]])

        return candidate_pts
    
        #  CONVEX FRONTIER LOGIC:      WORKING
        #         if(occupancy[potential_point[0],potential_point[1]] == 0):
        #             pts.append([potential_point[0],potential_point[1]])
        #             #print("In free space and ",occupancy[potential_point[0],potential_point[1]])
        #         else:
        #             break_flag = 0
        #             #print("Point Not In Free Space")
        #             for i in range(1,20):
        #                 if break_flag == 1:
        #                     break
        #                 for j in range(potential_point[0]-i,potential_point[0]+i+1):
        #                     if break_flag == 1:
        #                         break
        #                     for k in range(potential_point[1]-i,potential_point[1]+i+1):
        #                         if j == potential_point[0] & k == potential_point[1]:
        #                             pass
        #                         else:
        #                             if occupancy[j,k] == 0:
        #                                 pts.append([j,k])
        #                                 break_flag=1
        #                                 break

        # return pts 

    def select_point(self, candidate_points, occupancy_map, nearest = True):
        '''
        Selects a single point from the list of potential points using IG
        
        Inputs:
        candidate_points -> Candidate Points: list -> [(x, y), ...]
        occupancy_map -> Occupancy Grid
        nearest -> True: use nearest priority, False: use entropy
        
        Output: 
        candidate_points_ordered: Candidate Points orders according to IG 
        ''' 

        # Map prepocessing, to have probabilities instead of absolute values
        # Map the absolute values to probability of that cell being occupied
        occupancy_map = occupancy_map.astype(np.float32)
        occupancy_map[np.where(occupancy_map==-1.0)] = 0.5 # Unknown space
        occupancy_map[np.where(occupancy_map==0.0)] = 0.0001 # Free space
        occupancy_map[np.where(occupancy_map==100.0)] = 1.0 # Occupied space

        # List of entropies of candidate points
        IG = []
        distances = []

        # mask size of entropy summation
        mask_size = int(10/2)

        # Iterate over each candidate point and compute its information gain
        for r, c in candidate_points:
            
            # Compute entropy in the neighborhood of that cell
            # Stores neighboring cell entropies
            if nearest:  # Use distance
                # distacne between pose (real base) and candidate pt (grid base)
                d = self.pose_to_grid_distance([r,c], self.current_pose[0:2])
                distances.append(d)
                # print('candidate_points list: ', candidate_points)
                # print('distances list: ', distances)
            else:       # use IG based on entropy
                neighbor_prob = []
                for i in range(r-mask_size,r+mask_size+1):
                    for j in range(c-mask_size,c+mask_size+1):
                        neighbor_prob.append(occupancy_map[i,j])

                neighbor_prob = np.array(neighbor_prob)

                # Transform these probabilities to entropies: See associated thesis chapter ()
                entropy = -(neighbor_prob*np.log(neighbor_prob) + (1.0-neighbor_prob)*np.log(1-neighbor_prob))
                entropy = np.nan_to_num(entropy)
                # Sum the entropies in neighborhood
                abs_entropy = np.sum(entropy)
                # add to candidate point entropies list
                IG.append(abs_entropy)

        # Minimize distance
        if nearest:  
            # Candidate points are ordered according to their closeness to the robot
            candidate_points_ordered = []
            while candidate_points!=[]:    
                idx = distances.index(min(distances))
                distances.pop(idx)
                candidate_points_ordered.append(candidate_points.pop(idx))

        # Maximize distance
        else:  
            # Candidate points are now ordered according to their information gains, so according to their priority
            idx = IG.index(max(IG))
            candidate_points_ordered = []
            while candidate_points!=[]:    
                idx = IG.index(max(IG))
                IG.pop(idx)
                candidate_points_ordered.append(candidate_points.pop(idx))

        return np.array(candidate_points_ordered)

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

    def pose_to_grid_distance(self,grid, pose):
        return np.sqrt(np.sum((np.array(pose) - np.array(self.__map_to_position__(grid)))**2))

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
    n = Frontier_explorer()
    rospy.spin()
