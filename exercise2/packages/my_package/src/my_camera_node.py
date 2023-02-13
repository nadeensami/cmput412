#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from PIL import Image
import io

'''
Basic code for a node was taken from
Unit C-2: Development in the Duckietown infrastructure, Hands-on Robotics Development using Duckietown
Link: https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
'''
class MyCameraNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyCameraNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher and subscriber
        self.sub = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed", CompressedImage, self.callback)
        self.pub = rospy.Publisher('custom_camera/compressed', CompressedImage, queue_size=10)

        self.image = None

    def callback(self, data):
        image = Image.open(io.BytesIO(data.data))
        width = image.width
        height = image.height
        rospy.loginfo(f"Image width is {width} and height is {height}")
        self.image = data

    def run(self):
        # publish message every 2 seconds
        rate = rospy.Rate(0.5) # 0.5Hz
        while not rospy.is_shutdown():
            rospy.loginfo("Publishing an image")
            if self.image:
                self.pub.publish(self.image)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = MyCameraNode(node_name='my_camera_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()