#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String

'''
Basic code for a node was taken from
Unit C-2: Development in the Duckietown infrastructure, Hands-on Robotics Development using Duckietown
Link: https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
'''
class MySubscriberNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.sub = rospy.Subscriber('~chatter', String, self.callback)

    def callback(self, data):
        rospy.loginfo("I heard %s", data.data)

if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='my_subscriber_node')
    # keep spinning
    rospy.spin()