Nada Abbas - Khawaja Ghulam Alamdar

pick_up_objects_task - April 1, 2023

# How to use:

- Step (1): Unzip the pick\_up\_objects\_task package in your catkin workspace.

- Step (2): Execute the following commands:

    export TURTLEBOT3\_MODEL=burger

    roslaunch pick\_up\_objects\_task pick\_up\_objects\_task.launch

    rosrun pick\_up\_objects\_task pickup\_behaviors.py

# Overview:

The Figure below shows the behaviour tree used for this lab.

![](media/mobile_pick_n_place.png) 

Three blackboard variables are used. object name records the can type, current pickup keeps a track of which pickup location should be visited next and n cans keeps a track of number of cans successfully dropped. The program terminates when both cans have been dropped or when all pickup locations have been visited.

# Challenges Faced:

## Multi-window Neighborhood Approach

Most issues faced were with regards to the big can (beer). This can is large enough to be visible in the occupancy grid as an obstacle. When planning, ompl takes care that even if the goal is an obstacle, it selects the nearest possible reachable point, according to the distance threshold. This is ﬁne until the robot scan starts seeing more parts of the can and its size on the map increases. It also happens due to a few ﬂickering cells in the map. This causes the check path function to return invalid, and the path is recomputed. It turns into a continuous loop, and the robot is never able to reach the end goal because the path keeps changing. To solve this, two neighborhood windows are used for the is valid function. The given distance threshold is used for the neighborhood used in determining the validity of path when computing it. But when checking if the path is still valid, a smaller window size is used. This ensures that the re-planning does not happen continuously because of the obstacle expanding slightly or a few obstacle cells ﬂickering.

## Landing on Obstacle Cells

As a consequence of the previous solution, another issue is pronounced. Since the robot is now going too close to the obstacle space of the big can, it often happens that after picking the can it is too close to the obstacle cells or on it. Since the map is not instantly updated when the can is picked and because the robot is above the obstacle space sometimes, the obstacle cells remain obstacle instead of becoming free. This invalidates the path both in computing and checking. For check path, a simple solution can be implemented. Just ignore the ﬁrst few way-points (distance threshold + a safety factor (1.2\*distance threshold)) since these are the points most likely to be on obstacle cells. Additionally, last few waypoints are also ignored to aid with the previous issue. This, however, does not help in the compute path functionality, as ompl still ﬁnds the current robot pose on/near obstacle cells, and returns no valid path. To solve this, a rudimentary approach is used. The current start location is oﬀset by a constant value and passed to compute path function instead of the current pose. This oﬀset keeps getting compounded until a solution is found. In this case, the ﬁrst waypoint is not removed from the path list, as the ﬁrst element is no longer the current robot pose, but a nearby free cell.

## Double Initialization of Move Behaviour

The Move behaviours contain the information about pickup and drop points. Initially, it also contained blackboard variable initializations. The problem, however, was that the Move behaviour is instantiated twice, ﬁrst as the ﬁrst child responsible for pickup, and then as the fourth child responsible for drop-oﬀ. All blackboard variables are thus initialized in the check objects behaviour.


## Additional Comments

Since the environment is more constrained than lab 1, the smaller turtlebot model Burger is used instead of Waﬄe, and the distance threshold is adjusted according to its width plus a safety factor.

Another corner case is encountered when the robot is on its way to drop the bigger can, and the smaller can lies on its path. This smaller can is not visible on map as it’s too small. A very small map resolution was tested but the can was still not visible. Changing the model of the smaller can to bigger can solved this issue. However, for the cans to be distinguishable, the submitted package does not reﬂect this change.

<p float="left">
  <img src="media/can%20on%20path.png" width="500" />
  <img src="media/hit_can.png" width="300" /> 
</p>

**Figure: Smaller Can on Path**

A issue that remained is that while the robot is carrying a can, the movements are a bit jerky. It is suspected that it is likely because of some collision conﬂicts between the robot and can model in Gazebo. Since this issue is not directly related to the lab task, it is ignored.


# Examples of use:

<p float="left">
  <img src="media/full1-1.png" width="300" />
  <img src="media/full1-2.png" width="300" /> 
</p>

**Figure: Example 1**

<p float="left">
  <img src="media/full3-2.png" width="300" />
  <img src="media/full3-3.png" width="300" /> 
</p>

**Figure: Example 2**

# Media

Kindly, ﬁnd the attached drive link for videos [here](https://drive.google.com/drive/folders/1LUQ3a9xyeWg5Wo-S24KDBxm9-e1aLSgl?usp=sharing).


