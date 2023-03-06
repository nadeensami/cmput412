#!/usr/bin/env python3
import rospy, cv2, tf
import numpy as np

from duckietown.dtros import DTROS, NodeType, TopicType
from cv_bridge import CvBridge
from dt_apriltags import Detector
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer

from duckietown_msgs.msg import LEDPattern, AprilTagDetectionArray, AprilTagDetection
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Transform, Vector3, Quaternion

"""
  Much of the code for this apriltag detection class was taken from
  apriltag_detector_node.py, in duckietown/dt-core Github repo
  Authors: afdaniele, CourchesneA, AndreaCensi
  Link: https://github.com/duckietown/dt-core/blob/daffy/packages/apriltag/src/apriltag_detector_node.py
"""
class ARNode(DTROS):
  def __init__(self, node_name):
    # Initialize the DTROS parent class
    super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
    self.veh_name = rospy.get_namespace().strip("/")

    # Get static parameters
    self.bridge = CvBridge()
    self.tag_size = 0.065
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

    # Initialize LED color-changing
    self.color_publisher = rospy.Publisher(f'/{self.veh_name}/led_emitter_node/led_pattern', LEDPattern, queue_size = 1)
    self.pattern = LEDPattern()
    self.pattern.header = Header()

    # Map of April tag id to color
    self.curr_id = 'none'
    self.last_id = 'none'
    self.id_color_map = {
      '169': {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}, # Stop - Red
      '162': {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}, # Stop - Red
      '153': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0}, # T-intersection - Blue
      '133': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0}, # T-intersection - Blue
      '62': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0}, # T-intersection - Blue
      '58': {'r': 0.0, 'g': 0.0, 'b': 1.0, 'a': 1.0}, # T-intersection - Blue
      '94': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}, # UofA tag - Green
      '93': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}, # UofA tag - Green
      '201': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}, # UofA tag - Green
      '200': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0}, # UofA tag - Green
      'other': {'r': 1.0, 'g': 1.0, 'b': 0.0, 'a': 0.0}, # Other tag - Yellow
      'none': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0} # No detections - white
    }

    # Create a CV bridge object
    self._jpeg = TurboJPEG()

    # Initialize detector
    self.at_detector = Detector(
      searchpath = ['apriltags'],
      families = 'tag36h11',
      nthreads = 1,
      quad_decimate = 1.0,
      quad_sigma = 0.0,
      refine_edges = 1,
      decode_sharpening = 0.25,
      debug = 0
    )

    # Initialize broadcaster
    self._tf_bcaster = tf.TransformBroadcaster()

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

    # Timer
    self.publish_hz = 1
    self.timer = rospy.Timer(rospy.Duration(1 / self.publish_hz), self.cb_timer)
    self.last_message = None

    # Location publisher
    self.location_pub = rospy.Publisher("location", Transform, queue_size=1)

    # Transform listener
    self.listener = tf.TransformListener(True, rospy.Duration(10.0))

    self.transform_publish_hz = 0.5
    self.timer = rospy.Timer(rospy.Duration(1 / self.transform_publish_hz), self.apply_transform)
    self.min_tag_id = None
  
  def cb_timer(self, _):
    '''
    Callback for timer
    '''
    msg = self.last_message
    if not msg:
      return

    # turn image message into grayscale image
    img = self._jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
    # run input image through the rectification map
    img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)
    # detect tags
    tags = self.at_detector.detect(img, True, self._camera_parameters, self.tag_size)
    # pack detections into a message
    tags_msg = AprilTagDetectionArray()
    tags_msg.header.stamp = msg.header.stamp
    tags_msg.header.frame_id = msg.header.frame_id
    min_tag_distance = float('inf')
    for tag in tags:
      # turn rotation matrix into quaternion
      q = _matrix_to_quaternion(tag.pose_R)
      p = tag.pose_t.T[0]
      # create single tag detection object
      detection = AprilTagDetection(
        transform=Transform(
          translation=Vector3(x=p[0], y=p[1], z=p[2]),
          rotation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        ),
        tag_id=tag.tag_id,
        tag_family=str(tag.tag_family),
        hamming=tag.hamming,
        decision_margin=tag.decision_margin,
        homography=tag.homography.flatten().astype(np.float32).tolist(),
        center=tag.center.tolist(),
        corners=tag.corners.flatten().tolist(),
        pose_error=tag.pose_err,
      )
      # publish tf
      self._tf_bcaster.sendTransform(
        p.tolist(),
        q.tolist(),
        msg.header.stamp,
        "tag/{:s}".format(str(tag.tag_id)),
        msg.header.frame_id,
      )
      distance = tag.pose_t[2][0]
      if distance < min_tag_distance:
        self.min_tag_id = tag.tag_id
        min_tag_distance = distance
      # add detection to array
      tags_msg.detections.append(detection)
    
    # render visualization (if needed)
    self._render_detections(msg, img, tags)

  def apply_transform(self, _):
    if not self.min_tag_id:
      return
    tag_id = self.min_tag_id
    try:
      (intermediate_translation, intermediate_rotation) = self.listener.lookupTransformFull(
        f"tag/{str(tag_id)}", 
        rospy.Time(0),
        "odometry",
        rospy.Time(0),
        "world"
      )

      self._tf_bcaster.sendTransform(
        intermediate_translation,
        intermediate_rotation,
        rospy.Time.now(),
        f"intermediate_tag/{str(tag_id)}",
        f"at_{str(tag_id)}_static",
      )

      (translation, rotation) = self.listener.lookupTransformFull(
        "world",
        rospy.Time(0),
        f"intermediate_tag/{str(tag_id)}",
        rospy.Time(0),
        f"at_{str(tag_id)}_static",
      )
      
      transform = Transform(
        translation=Vector3(x=translation[0], y=translation[1], z=translation[2]),
        rotation=Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3]),
      )

      self.location_pub.publish(transform)
    except Exception as e:
      print(e)
      return
  
  def image_cb(self, msg):
    '''
    Callback for compressed camera images
    '''
    self.last_message = msg
    
  def _render_detections(self, msg, img, detections):
    # get a color buffer from the BW image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # if we have no detections
    if len(detections) == 0:
      self.curr_id = 'none'
    # draw each tag
    for detection in detections:
      for idx in range(len(detection.corners)):
        cv2.line(
          img,
          tuple(detection.corners[idx - 1, :].astype(int)),
          tuple(detection.corners[idx, :].astype(int)),
          (0, 255, 0),
        )
      # draw the tag ID
      cv2.putText(
        img,
        str(detection.tag_id),
        org=(detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) + 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 255),
      )
      if str(detection.tag_id) in self.id_color_map:
        self.curr_id = str(detection.tag_id)
      else:
        self.curr_id = 'other'
    
    # pack image into a message
    img_msg = CompressedImage()
    img_msg.header.stamp = msg.header.stamp
    img_msg.header.frame_id = msg.header.frame_id
    img_msg.format = "jpeg"
    img_msg.data = self._jpeg.encode(img)

    # publish image
    self.image_pub.publish(img_msg)

  def onShutdown(self):
    super(ARNode, self).onShutdown()

  def change_color(self, color):
    '''
    Code for this function was inspired by 
    "duckietown/dt-core", file "led_emitter_node.py"
    Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
    Author: GitHub user liampaull
    '''
    self.pattern.header.stamp = rospy.Time.now()
    rgba = ColorRGBA()
    rgba.r = color['r']
    rgba.g = color['g']
    rgba.b = color['b']
    rgba.a = color['a']
    self.pattern.rgb_vals = [rgba] * 5
    self.color_publisher.publish(self.pattern)
  
  def run(self):
    # Start at no detections
    self.change_color(self.id_color_map['none'])

    rate = rospy.Rate(10) # 10 times a second

    while not rospy.is_shutdown():
      if self.curr_id != self.last_id:
        self.change_color(self.id_color_map[self.curr_id])
        self.curr_id = self.last_id
      rate.sleep()

def _matrix_to_quaternion(r):
  T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), dtype=np.float64)
  T[0:3, 0:3] = r
  return tf.transformations.quaternion_from_matrix(T)

if __name__ == '__main__':
  # Initialize the node
  ar_node = ARNode(node_name='augmented_reality_apriltag_node')
  # Run LED task
  # ar_node.run()
  # Keep it spinning to keep the node alive
  rospy.spin()