#!/usr/bin/env python3
import rospy
# from my_package.srv import ColorServiceResponse, ColorService
# from std_msgs.msg import Bool, String
# from std_srvs.srv import SetBoolResponse
from duckietown_msgs.srv import SetFSMState
from duckietown.dtros import DTROS, NodeType

NAME = 'led_server'
'''
Basic code for a client was taken from
Writing a Simple Service and Client (Python), ROS Wiki
Link: http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29

Basic code for a node was taken from
Unit C-2: Development in the Duckietown infrastructure, Hands-on Robotics Development using Duckietown
Link: https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
'''
class LEDClientNode(DTROS):
  def __init__(self, node_name):
    super(LEDClientNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

    rospy.wait_for_service(NAME)

    colors = ['red', 'green', 'blue', 'white']

    try:
      change_color = rospy.ServiceProxy(NAME, SetFSMState)


      run = True
      while run:
        color = input("Please select your color:")
        if color == '':
          run = False
        else:
          rospy.loginfo("Sending color: %s", color)
          change_color(color)

      # rate = rospy.Rate(0.5)
      # for color in colors:
      #   rospy.loginfo("Sending color: %s", color)
      #   change_color(color)
      #   rate.sleep()

    except rospy.ServiceException as e:
      print("Service call failed: %s" % e)

if __name__ == '__main__':
  # Create the LEDClientNode object
  led_emitter_node = LEDClientNode(node_name="led_client_node")
  # Keep it spinning to keep the node alive
  rospy.spin()