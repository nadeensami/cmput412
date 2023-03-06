#!/usr/bin/env python3
import rospy, cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType, TopicType
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
from image_geometry import PinholeCameraModel

from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Header

"""
  Much of the code for this lane following class was taken from
  Turtlebot Line Follower, in Gaitech docs
  Link: http://edu.gaitech.hk/turtlebot/line-follower.html
"""
class LaneFollowingNode(DTROS):
  def __init__(self, node_name):
    # Initialize the DTROS parent class
    super(LaneFollowingNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
    self.veh_name = rospy.get_namespace().strip("/")
    rospy.on_shutdown(self.onShutdown)
    self.is_shutdown = False

    # Get static parameters
    self.bridge = CvBridge()
    self.rectify_alpha = 0.0
    
    # Initialize static parameters from camera info message
    camera_info_msg = rospy.wait_for_message(f'/{self.veh_name}/camera_node/camera_info', CameraInfo)
    self.camera_model = PinholeCameraModel()
    self.camera_model.fromCameraInfo(camera_info_msg)
    H, W = camera_info_msg.height, camera_info_msg.width
    # find optimal rectified pinhole camera
    rect_K, _ = cv2.getOptimalNewCameraMatrix(
      self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
    )
    # store new camera parameters
    self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])
    # create rectification map
    self._mapx, self._mapy = cv2.initUndistortRectifyMap(
      self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
    )

    # Create a CV bridge object
    self._jpeg = TurboJPEG()

    # Subscriber
    self.image_sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.image_cb, queue_size = 1)

    # Publisher
    self.image_pub = rospy.Publisher(
      f"/{self.veh_name}/detections/image/compressed",
      CompressedImage,
      queue_size = 1,
      dt_topic_type = TopicType.VISUALIZATION,
      dt_help = "Camera image with tags superimposed",
    )
    self.cmd_vel_pub = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size = 1, dt_topic_type = TopicType.CONTROL)

    self.twist = Twist2DStamped()
    self.twist.header = Header()
  
  def image_cb(self, msg):
    '''
    Callback for compressed camera images
    '''
    if self.is_shutdown:
      return
    
    image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Observed hsv datapoints
    # hsv(54, 44%, 72%)
    # hsv(56, 69%, 84%)
    # hsv(56, 95%, 85%)
    # hsv(57, 96%, 86%)
    lower_yellow = np.array([20, 40, 60])
    upper_yellow = np.array([70, 120, 100])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    maskedFrame = cv2.bitwise_and(image, image, mask = mask)

    h, w, d = image.shape
    search_top = int(3*h/4)
    search_bot = int(3*h/4 + 20)
    mask[0:search_top, 0:w] = 0
    mask[search_bot:h, 0:w] = 0

    M = cv2.moments(mask)
    if M['m00'] > 0:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)
      # The proportional controller is implemented in the following four lines which
      # is responsible of linear scaling of an error to drive the control output.

      err = cx - w/2
      self.twist.header.stamp = msg.header.stamp
      self.twist.header.frame_id = msg.header.frame_id
      self.twist.v = 0.3
      self.twist.omega = -float(err) / 100
      self.cmd_vel_pub.publish(self.twist)
    
    img_msg = CompressedImage()
    img_msg.header.stamp = msg.header.stamp
    img_msg.header.frame_id = msg.header.frame_id
    img_msg.format = "jpeg"
    img_msg.data = self._jpeg.encode(maskedFrame)

    # publish image
    self.image_pub.publish(img_msg)

  def onShutdown(self):
    self.is_shutdown = True

    # Stop twisting
    self.twist.v = 0
    self.twist.omega = 0
    self.cmd_vel_pub.publish(self.twist)

if __name__ == '__main__':
  # Initialize the node
  ar_node = LaneFollowingNode(node_name='lane_following_node')
  # Keep it spinning to keep the node alive
  rospy.spin()