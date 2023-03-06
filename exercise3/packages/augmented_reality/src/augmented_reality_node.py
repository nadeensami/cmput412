#!/usr/bin/env python3

import yaml, cv2, rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge

'''
Basic code for a node was taken from
Unit C-2: Development in the Duckietown infrastructure, Hands-on Robotics Development using Duckietown
Link: https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
'''
class AugmentedRealityNode(DTROS):
  def __init__(self, node_name):
    # Initialize the DTROS parent class
    super(AugmentedRealityNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.veh_name = rospy.get_namespace().strip("/")

    # Get static parameters
    calibration_file = f'/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml'
    self.homography = self.readYamlFile(calibration_file)['homography']
    self.camera_info_msg = rospy.wait_for_message(f'/{self.veh_name}/camera_node/camera_info', CameraInfo)

    # Get points and segments from map file
    self._points = rospy.get_param('~points')
    self._segments = rospy.get_param('~segments')

    # TODO: get map file base name
    map_file_base_name = 'hud'

    # Initialize Augmenter
    self.augmenter = Augmenter(self.homography, self.camera_info_msg)

    # Subscriber
    self.sub_image = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.project, queue_size = 1)

    # Publisher
    self.pub_result = rospy.Publisher(f'/{self.veh_name}/{node_name}/{map_file_base_name}/image/compressed', CompressedImage, queue_size = 1)
    
  def readYamlFile(self, fname):
    """
    Reads the YAML file in the path specified by 'fname'.
    E.G. :
        the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
    """
    with open(fname, 'r') as in_file:
      try:
        yaml_dict = yaml.load(in_file)
        return yaml_dict
      except yaml.YAMLError as exc:
        rospy.loginfo("YAML syntax error. File: %s fname. Exc: %s"
          %(fname, exc), type='fatal')
        rospy.signal_shutdown()
        return
      
  def project(self,msg):
    br = CvBridge()
    self.raw_image = br.compressed_imgmsg_to_cv2(msg)
    dis = self.augmenter.process_image(self.raw_image)
    render = self.augmenter.render_segments(points=self._points, img=dis, segments=self._segments)
    result = br.cv2_to_compressed_imgmsg(render,dst_format='jpg')
    self.pub_result.publish(result)

'''
Augmenter class was taken from
GitHub Repository exA-3 by GitHub user Coral79
Link: https://github.com/Coral79/exA-3/blob/44adf94bad728507608086b91fbf5645fc22555f/packages/augmented_reality_basics/include/augmented_reality_basics/augmented_reality_basics.py
''' 
class Augmenter():
  def __init__(self, homography, camera_info_msg):
    self.H = [homography[0:3], homography[3:6], homography[6:9]]
    self.Hinv = np.linalg.inv(self.H)
    self.K = np.array(camera_info_msg.K).reshape((3, 3))
    self.R = np.array(camera_info_msg.R).reshape((3, 3))
    self.D = np.array(camera_info_msg.D[0:4])
    self.P = np.array(camera_info_msg.P).reshape((3, 4))
    self.h = camera_info_msg.height
    self.w = camera_info_msg.width

  def process_image(self, cv_image_raw):
    '''
    Undistort an image.
    '''
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
    res = cv2.undistort(cv_image_raw, self.K, self.D, None, newcameramtx)
    return res

  def ground2pixel(self, point):
    '''
    Transforms points in ground coordinates (i.e. the robot reference frame) to pixels in the image.
    '''
    if len(point) > 2 and point[2] != 0:
      msg = 'This method assumes that the point is a ground point (z=0).'
      msg += 'However, the point is (%s,%s,%s)' % (point.x, point.y, point.z)
      raise ValueError(msg)

    ground_point = np.array([point[0], point[1], 1.0])
    image_point = np.dot(self.Hinv, ground_point)
    image_point = image_point / image_point[2]

    pixel = image_point[0:2]
    pixel = np.round(pixel).astype(int)
    return pixel

  def render_segments(self, points, img, segments):
    for i in range(len(segments)):
      point_x = points[segments[i]["points"][0]][1]
      point_y = points[segments[i]["points"][1]][1]
      point_x = self.ground2pixel(point_x)
      point_y = self.ground2pixel(point_y)
      color = segments[i]["color"]
      img = self.draw_segment(img, point_x, point_y, color)
    return img

  def draw_segment(self, image, pt_x, pt_y, color):
    defined_colors = {
      'red': ['rgb', [1, 0, 0]],
      'green': ['rgb', [0, 1, 0]],
      'blue': ['rgb', [0, 0, 1]],
      'yellow': ['rgb', [1, 1, 0]],
      'magenta': ['rgb', [1, 0, 1]],
      'cyan': ['rgb', [0, 1, 1]],
      'white': ['rgb', [1, 1, 1]],
      'black': ['rgb', [0, 0, 0]]}
    _, [r, g, b] = defined_colors[color]
    cv2.line(image, (pt_x[0], pt_x[1]), (pt_y[0], pt_y[1]), (b * 255, g * 255, r * 255), 5)
    return image

if __name__ == '__main__':
  # create the node
  node = AugmentedRealityNode(node_name='augmented_reality_node')
  # keep spinning
  rospy.spin()