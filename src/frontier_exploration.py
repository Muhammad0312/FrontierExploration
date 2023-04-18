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

        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_yaw = 0.0

        self.drone_u = 0.0
        self.drone_v = 0.0
        
        self.myMarker = Marker()
        self.myMarker.header.frame_id = "odom"
        self.myMarker.header.stamp  = rospy.get_rostime()
        self.motion_busy = None
        #self.myMarker.ns = "window"
        self.myMarker.id = 1
        self.myMarker.type = self.myMarker.SPHERE # sphere
        self.myMarker.action = self.myMarker.ADD

        self.myPoint = Point()
        self.myPoint.z = 1.5
        self.myMarker.pose.position = self.myPoint
        self.myMarker.color=ColorRGBA(0, 1, 0, 1)
        self.myMarker.scale.x = 0.2
        self.myMarker.scale.y = 0.2
        self.myMarker.scale.z = 0.2
        self.myMarker.lifetime = rospy.Duration(0)
        
        self.marker_Arr = MarkerArray()
        self.marker_Arr.markers = []

        #
        # self.frontier_status_publisher = rospy.Publisher("/frontier/status",Int32MultiArray,queue_size=1) #occupancy grid publisher
        
        # self.frontier_grid_goal_publisher = rospy.Publisher("/frontier/grid_goal",Int32MultiArray,queue_size=1) #occupancy grid publisher
        
        self.grid_pos_publisher = rospy.Publisher("/frontier/grid_pos",Int32MultiArray,queue_size=1) #occupancy grid publisher
        
        # self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)
        
        # Publish list of candidate points in order of preference
        self.frontier_grid_goal_list = rospy.Publisher("/frontier/grid_goal_list",Int32MultiArray,queue_size=1) #occupancy grid publisher

        # self.occupancy_filtered_grid_publisher = rospy.Publisher("/frontier/filtered_occupancy",OccupancyGrid,queue_size=1) #occupancy grid publisher
        
        # Frontier Points for Visualization
        self.frontier_points_pub = rospy.Publisher("/frontier_detection/vis_points",MarkerArray,queue_size=1) #occupancy grid publisher

        self.map_subscriber = rospy.Subscriber("/projected_map", OccupancyGrid, self.projected_map_callback) #/rtabmap/grid_map.
        # self.drone_pose = rospy.Subscriber("/gazebo/model_states", ModelStates, self.__pose_callback)
        
        # robot odometry/pose
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.get_odom)

    '''
    Robot odometry/pose callback
    '''
    def get_odom(self, odom):
        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])

        self.drone_x = odom.pose.pose.position.x
        self.drone_y = odom.pose.pose.position.y
        self.drone_yaw = yaw

        self.drone_v = int((self.drone_x - self.map_origin[0])/self.map_resolution)
        self.drone_u = int((self.drone_y - self.map_origin[1])/self.map_resolution)

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
        return lst

    def projected_map_callback(self, data):
        '''
        Called when a new occupancy grid is received. The function first removes noise from 
        the new occupancy i.e. dilates obstacles, and then publishes the filtered occupancy. 
        Then a frontier/points are identified on the map. A candidate point is selected 
        and published under topic '/frontier/grid_goal'. 
        '''

        #meta deta of map: 
        map_r1 = data.info.height  #y
        map_c1 = data.info.width   #x
        self.map_origin = [data.info.origin.position.x, data.info.origin.position.y]  # origin
        self.map_resolution = data.info.resolution
        map_t = [ self.map_origin[0] + map_c1*self.map_resolution, self.map_origin[1] + map_r1*self.map_resolution]

        #read map: 
        occupancy_map = np.array(data.data).reshape(map_r1,map_c1)
        (self.r, self.c) = occupancy_map.shape 
	
        #identify frontiers: Binarization
        frontiers = self.identify_frontiers(occupancy_map)        

	    # label the diff frontiers: Segmentation
        candidate_pts = self.label_frontiers(frontiers,occupancy_map)         
    
	    # Select Points: Candidate Point Selection
        candidate_pts_ordered = self.select_point(candidate_pts, occupancy_map)
    
        #frontier_send = Int32MultiArray()
        pose_send = Int32MultiArray()
    
        #frontier_send.data = selected_point
        pose_send.data = [self.drone_v,self.drone_u]
    
        #self.frontier_grid_goal_publisher.publish(frontier_send)
        self.grid_pos_publisher.publish(pose_send)
    
        general_pts = self.__all_map_to_position__(candidate_pts_ordered)
        # self.publish_path(general_pts)
        self.publish_frontier_points(general_pts)
        
        # candidate_pts_ordered = self.flatten_array(candidate_pts_ordered)

        candidate_pts_ordered_send = Int32MultiArray()
        candidate_pts_ordered_send.data = candidate_pts_ordered
        #print('candidate pts: ', candidate_pts_ordered_send)
        self.frontier_grid_goal_list.publish(candidate_pts_ordered_send)


    def identify_frontiers(self, octomap):
        '''
        Identifies frontiers i.e. region between free space and unknown space. 
        Input: 
        octomap -> occupancy grid with 3 values [0, 50, 100]
        
        Output:
        result -> occupancy grid with frontier cells with value 255 (any max value) 
        '''
        mask_size = 3 #(3, 3)
        result = np.zeros(shape = (self.r, self.c))
        ##print('unique values:', np.unique(octomap))
        for i in range(7, self.r-7):
            for j in range(7, self.c-7): 

                #apply mask: 
                flag = 0
                for a in range(-1, 1):
                    for b in range(-1, 1):
                        #filter calculations: 
                        if octomap[i, j] == -1 and octomap[i+a, j+b] == 0:
                            result[i, j] = 255
                            flag = 1
                        elif octomap[i, j] == 0 and octomap[i+a, j+b] == -1:
                            result[i, j] = 255
                            flag = 1
                if flag==0:
                    result[i, j] = 0
        demo_map = copy.deepcopy(result)
        cv2.imwrite("frontiers.png",demo_map)
                        
        return result

    def label_frontiers(self,frontiers,occupancy):
        labelled_frontiers = measure.label(frontiers, background=0)
        #print("Unique Frontiers: ",  np.unique(labelled_frontiers))
        regions = measure.regionprops(labelled_frontiers)
        #print('-----------Region Props--------------')
        pts = []
        #print(np.unique(frontiers))
        grid = copy.deepcopy(occupancy)
        occu = copy.deepcopy(grid)
        grid[np.where(grid==-1)]=127 # Unknown space
        grid[np.where(grid==0)]=255 # free space
        grid[np.where(grid==100)]=0 # occupied space
        cv2.imwrite("testing.png",grid)
        grid = cv2.imread("testing.png")
        for prop in regions:
            if prop.area > 5.0:
                orientation = prop.orientation
                #minor axis
                x0 = int(prop.centroid[0])
                y0 = int(prop.centroid[1])
                x1 = int(x0 + math.cos(orientation) * 0.5 * 15.0)
                y1 = int(y0 - math.sin(orientation) * 0.5 * 15.0)
                grid = cv2.circle(grid, (int(y0),int(x0)), 2, (255,0,0),2)
                rr, cc = draw.line(x0, y0, x1, y1)   # (r0, c0, r1, c1)

                # Both directions
                x2 = int(x0 - math.cos(orientation) * 0.5 * 15.0)
                y2 = int(y0 + math.sin(orientation) * 0.5 * 15.0)		

                if occu[x1,y1] == -1:
                        rr, cc = draw.line(x0, y0, x1, y1)   # (r0, c0, r1, c1)
                elif occu[x2,y2] == -1:
                        rr, cc = draw.line(x0, y0, x2, y2)   # (r0, c0, r1, c1)
                grid[rr, cc,0] = 255
                grid[rr, cc,1] = 0
                grid[rr, cc,2] = 0
                cv2.imwrite("frontiercenter.png",grid)
                
                ##print("unique filtered values: ",np.unique(occupancy))
                potential_point = [int(prop.centroid[0]),int(prop.centroid[1])] 
                if(occupancy[potential_point[0],potential_point[1]] == 0):
                    pts.append([potential_point[0],potential_point[1]])
                    #print("In free space and ",occupancy[potential_point[0],potential_point[1]])
                else:
                    break_flag = 0
                    #print("Point Not In Free Space")
                    for i in range(1,20):
                        if break_flag == 1:
                            break
                        for j in range(potential_point[0]-i,potential_point[0]+i+1):
                            if break_flag == 1:
                                break
                            for k in range(potential_point[1]-i,potential_point[1]+i+1):
                                if j == potential_point[0] & k == potential_point[1]:
                                    pass
                                else:
                                    if occupancy[j,k] == 0:
                                        pts.append([j,k])
                                        break_flag=1
                                        break

        return pts 

    def select_point(self, potential_points, occupancy_map):
        '''
        Selects a single point from the list of potential points
        inputs:
        potential_points: list -> [(x, y), ...]
        outputs: 
        int frontier_index 
        ''' 
        occupancy_map = occupancy_map.astype(np.float32)
        occupancy_map[np.where(occupancy_map==-1.0)] = 0.5 # Unknown space
        occupancy_map[np.where(occupancy_map==0.0)] = 0.0001# free space
        occupancy_map[np.where(occupancy_map==100.0)] = 1.0 # occupied space
        entropies = []

        #calculate entropy in local area (square): 
        self_entropy = []
         
        mask_size = 10
        for r, c in potential_points:

            local_map = []
            for i in range(r-5,r+5):
                lst_t = []
                for j in range(c-5,c+5):
                    lst_t.append(occupancy_map[i,j])
                local_map.append(lst_t)

            local_map = np.array(local_map)
            
            humans_in_vicinity = self.human_entropy((r, c))
            n = len(humans_in_vicinity)

            entropy = -(local_map*np.log(local_map) + (1.0-local_map)*np.log(1-local_map))
            entropy = np.nan_to_num(entropy)
            # #print(entropy)
            abs_entropy = np.sum(entropy) + n*mask_size

            #add to entropies list
            entropies.append(abs_entropy)
        
        idx = entropies.index(max(entropies))

        potential_points_ordered = []
        while potential_points!=[]:    
            idx = entropies.index(max(entropies))
            entropies.pop(idx)
            potential_points_ordered.append(potential_points.pop(idx))

        return potential_points_ordered

    def human_entropy(self, frontier_pt):
        #transform frontier point to global coordinates: 
        # frontier_pt = self.local_to_global(frontier_pt)

        # d = 1 #1m distance
        # humans_in_vicinity = []
        # for i in self.human_poses:
        #     if self.dist(frontier_pt, i) < d: 
        #         humans_in_vicinity.append(i)
        return []#humans_in_vicinity

    def dist(self, p1, p2):
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]

        return ((y2-y1)**2 + (x2-x1)**2)**(1/2)

    def local_to_global(self, p):
        x = -self.map_origin[1] - p[0]*self.map_resolution
        y = p[1]*self.map_resolution + self.map_origin[0]


    def callback2(self, data):
        
        pos = data.pose.position
        loc_x = pos.x
        loc_y = pos.y
        yaw = data.pose.orientation.z

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

if __name__ == '__main__':
    rospy.init_node('frontier_explorer', anonymous=True)
    n = Frontier_explorer()
    rospy.spin()
