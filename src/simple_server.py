#!/usr/bin/python3

import roslib
roslib.load_manifest('frontier_explorationb')
import rospy
import actionlib

from frontier_explorationb.msg import go_to_pointAction

class DoDishesServer:
  def __init__(self):
    self.server = actionlib.SimpleActionServer('do_dishes', go_to_pointAction, self.execute, False)
    self.server.start()

  def execute(self, goal):
    # Do lots of awesome groundbreaking robot stuff here
    self.server.set_succeeded()


if __name__ == '__main__':
  rospy.init_node('do_dishes_server')
  server = DoDishesServer()
  rospy.spin()