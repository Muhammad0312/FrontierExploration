import numpy as np
import math
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


class FrontierDetector:

    # Constructor
    def __init__(self):

        self.entropy = 0 
        self.r = 0 #width of the map
        self.c = 0 #height of the map

        self.map_origin = [0.0,0.0]
        self.map_resolution = 0.0
        self.occupancy_map = None

        self.robot_len = 0.22

        # Current robot pose [x, y, yaw], None if unknown            
        self.current_pose = None
        
        self.motion_busy = None   

    def set_mapNpose(self,data,pose):
        # Fetch meta deta of map: 
        map_r1 = data.info.height  #y
        map_c1 = data.info.width   #x
        self.map_origin = [data.info.origin.position.x, data.info.origin.position.y] 
        self.map_resolution = data.info.resolution
        (self.r, self.c) = (map_r1,map_c1)
        
        # stiore occupancu grid map: 
        self.occupancy_map = self.__dilate_map__(np.array(data.data).reshape(map_r1,map_c1))
    
        self.current_pose = pose

    def getCandidatePoint(self,criterion='entropy'):
        #identify frontiers: Binarization
        frontiers = self.identify_frontiers()        

	    # label the diff frontiers: Segmentation
        candidate_pts, labelled_frontiers = self.label_frontiers(frontiers)         
    
	    # Select Points: Candidate Point Selection
        if criterion=='entropy':
            nearest = False
        else: 
            nearest = True

        candidate_pts_ordered = self.select_point(candidate_pts, nearest)

        return candidate_pts_ordered, labelled_frontiers

    def identify_frontiers(self):
        '''
        Identifies frontiers i.e. region between free space and unknown space. 
        Input: 
        octomap -> occupancy grid [-1, 0, 100]
        
        Output:
        result -> Filtered occupancy grid with frontier cells with value 255 (any max value), others 0
        See media -> frontiersX.png for reference 
        '''

        octomap = copy.deepcopy(self.occupancy_map)

        # binary image: 255 for frontier cells, otherwise
        result = np.zeros(shape = (self.r, self.c))

        # iterate over cells: DONT avoid boundary cells 
        for i in range(1, self.r-1):
            for j in range(1, self.c-1):  
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

    def label_frontiers(self,frontiers):
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

        occupancy = copy.deepcopy(self.occupancy_map)

        # Output list of candidate points
        candidate_pts = []

        # Segment/cluster/label the frontiers
        # Background has 0 value, and each frontier has a unique ID (1,2,3)
        labelled_frontiers = measure.label(frontiers, background=0)
        
        #__________________________  IMAGE SAVING  _________________________________
        img_norm = cv2.normalize(labelled_frontiers, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the image to uint8 type
        label_img = img_norm.astype(np.uint8)
        
        # Apply a color map to the image
        color_img = cv2.applyColorMap(label_img, cv2.COLORMAP_HOT)
        cv2.imwrite("clustered.png",color_img)
        #______________________________________________________________________________
        
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

        return candidate_pts, labelled_frontiers

    def select_point(self, candidate_points, nearest = True):
        '''
        Selects a single point from the list of potential points using IG
        
        Inputs:
        candidate_points -> Candidate Points: list -> [(x, y), ...]
        occupancy_map -> Occupancy Grid
        nearest -> True: use nearest priority, False: use entropy
        
        Output: 
        candidate_points_ordered: Candidate Points orders according to IG 
        ''' 
        print(nearest)
        occupancy_map = copy.deepcopy(self.occupancy_map)

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
                        if self.__in_map__([i,j]):
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
    def all_map_to_position(self, pts):
        lst = []
        for p in pts:
            lst.append(self.__map_to_position__(p))
        return np.array(lst)

    def __in_map__(self, loc):
        '''
        loc: list of index [x,y]
        returns True if location is in map and false otherwise
        '''
        [mx,my] = loc
        if mx >= self.r-1 or my >= self.c-1 or mx < 0 or my < 0:
            return False 
        return True

    def __dilate_map__(self,occupancy_map):

        grid_size = int(self.robot_len/2 / 0.05)
        print(grid_size)
        octomap = copy.deepcopy(occupancy_map)
        print(self.r,self.c)

        for r in range(grid_size, self.r -grid_size):
            for c in range(grid_size, self.c-grid_size):
                
                for i in range(-grid_size, grid_size+1):
                    for j in range(-grid_size, grid_size+1):

                        if occupancy_map[r+i, c+j] == 100:
                            octomap[r,c] = 100
                            break

        print('obstacle before: ', np.count_nonzero(occupancy_map == 100))
        print('obstacle after: ', np.count_nonzero(octomap == 100))

        return octomap

    def pose_to_grid_distance(self,grid, pose):
        return np.sqrt(np.sum((np.array(pose) - np.array(self.__map_to_position__(grid)))**2))