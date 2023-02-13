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
class MyPublisherNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.pub = rospy.Publisher('~chatter', String, queue_size=10)

    def run(self):
        # publish message every 1 second
        rate = rospy.Rate(1) # 1Hz
        while not rospy.is_shutdown():
            message = "Hello from %s" % os.environ['VEHICLE_NAME']
            rospy.loginfo("Publishing message: '%s'" % message)
            self.pub.publish(message)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = MyPublisherNode(node_name='my_publisher_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()