#!/usr/bin/python3

import roslib
roslib.load_manifest('frontier_explorationb')
import rospy
import actionlib

from frontier_explorationb.msg import go_to_pointAction, go_to_pointGoal

if __name__ == '__main__':
    rospy.init_node('do_dishes_client')
    client = actionlib.SimpleActionClient('do_dishes', go_to_pointAction)
    client.wait_for_server()

    goal = go_to_pointGoal()
    # Fill in the goal here
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))