#!/usr/bin/env python3
import rospy
from duckietown_msgs.srv import SetFSMState, SetFSMStateResponse
from duckietown_msgs.msg import LEDPattern
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Header, ColorRGBA

NAME = 'led_server'

'''
Basic code for a server was taken from
Writing a Simple Service and Client (Python), ROS Wiki
Link: http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29
'''
class LEDServerNode(DTROS):
  def __init__(self, node_name):
    super(LEDServerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.service = rospy.Service(NAME, SetFSMState, self.change_color)
    self.veh_name = rospy.get_namespace().strip("/")

    self.publisher = rospy.Publisher(f'/{self.veh_name}/led_emitter_node/led_pattern', LEDPattern, queue_size=1)

    self.pattern = LEDPattern()
    self.pattern.header = Header()

    self.colors = {
      'purple': {'r': 1.0, 'g': 0.0, 'b': 1.0, 'a': 1.0},
      'blue': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0},
      'cyan': {'r': 0.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
      'red': {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0},
      'green': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0},
      'yellow': {'r': 1.0, 'g': 1.0, 'b': 0.0, 'a': 1.0},
      'white': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
      'off': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 0.0}
    }

  def change_color(self, request):
    '''
    Code for this function was inspired by 
    "duckietown/dt-core", file "led_emitter_node.py"
    Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
    Author: GitHub user liampaull
    '''
    color = request.state

    if color not in self.colors:
      return SetFSMStateResponse()

    self.pattern.header.stamp = rospy.Time.now()
    rgba = ColorRGBA()
    rgba.r = self.colors[color]['r']
    rgba.g = self.colors[color]['g']
    rgba.b = self.colors[color]['b']
    rgba.a = self.colors[color]['a']
    self.pattern.rgb_vals = [rgba] * 5
    self.publisher.publish(self.pattern)

    return SetFSMStateResponse()

if __name__ == '__main__':
  # Create the LEDServerNode object
  led_emitter_node = LEDServerNode(node_name="led_server_node")
  # Keep it spinning to keep the node alive
  rospy.spin()