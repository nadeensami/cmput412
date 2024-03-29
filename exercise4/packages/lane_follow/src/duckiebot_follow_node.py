#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Header, ColorRGBA, Float32, String
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from duckietown_msgs.srv import SetFSMState

DEBUG = True

class DuckiebotFollowNode(DTROS):
  def __init__(self, node_name):
    super(DuckiebotFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    # Subscribers
    self.distance_sub = rospy.Subscriber(
      f"/{self.veh}/duckiebot_distance_node/distance",
      Float32,
      self.cb_distance,
      queue_size=1,
      buff_size="20MB"
    )
    self.rotation_sub = rospy.Subscriber(
      f"/{self.veh}/duckiebot_distance_node/offset",
      Float32,
      self.cb_rotation,
      queue_size=1,
      buff_size="20MB"
    )
    self.distance_sub = rospy.Subscriber(
      f"/{self.veh}/front_center_tof_driver_node/range",
      Float32,
      self.cb_tof,
      queue_size=1,
      buff_size="20MB"
    )
    
    # Publishers
    self.vel_pub = rospy.Publisher(
      f"/{self.veh}/car_cmd_switch_node/cmd",
      Twist2DStamped,
      queue_size=1
    )
    self.color_publisher = rospy.Publisher(f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size = 1)
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.turn_on_light()

    # TOF variables
    self.range = None
    self.last_range_detected_time = None

    # Lane following service
    rospy.wait_for_service('lane_following_service')
    self.lane_follow = rospy.ServiceProxy('lane_following_service', SetFSMState)
    self.lane_following = False

    # Pose detection variables
    self.stale_time = 1
    rospy.Timer(rospy.Duration(1/5), self.stale_detection)
    self.last_distance_detected_time = None
    self.distance_from_robot = None
    self.last_rotation_detected_time = None
    self.rotation_of_robot = None

    self.velocity = 0.3
    self.twist = Twist2DStamped(v = 0, omega = 0)

    # Distance PID variables
    self.distance_proportional = None

    self.distance_P = 3 # offset velocity
    self.distance_D = -0.5
    self.last_distance_error = 0
    self.last_distance_time = rospy.get_time()

    # Angle PID variables
    self.angle_proportional = None

    self.angle_P = 0.005
    self.angle_D = -0.0004
    self.last_angle_error = 0
    self.last_angle_time = rospy.get_time()
    
    # Duckiebot-following variables
    self.following_distance = 0.3

    # Initialize LED color-changing
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.signalled = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

    self.loginfo("Initialized")
  
  def cb_distance(self, msg):
    self.distance_from_robot = msg.data
    self.distance_proportional = self.distance_from_robot - self.following_distance
    self.last_distance_detected_time = rospy.get_time()

  def cb_rotation(self, msg):
    self.rotation_of_robot = msg.data
    self.angle_proportional = self.rotation_of_robot
    self.last_rotation_detected_time = rospy.get_time()
  
  def cb_tof(self, msg):
    self.range = msg.range
    self.last_range_detected_time = rospy.get_time()

  def stale_detection(self, _):
    """
    Remove Duckiebot detections if they are longer than the stale time
    """
    if not self.last_distance_detected_time or not self.last_distance_detected_time \
    or (self.last_distance_detected_time and rospy.get_time() - self.last_distance_detected_time > self.stale_time) \
    or (self.last_rotation_detected_time and rospy.get_time() - self.last_rotation_detected_time > self.stale_time):
      self.distance_from_robot = None
      self.distance_proportional = None
      self.rotation_of_robot = None
      self.angle_proportional = None

      if not self.lane_following:
        self.lane_follow("True")
        self.lane_following = True
    elif (self.last_distance_detected_time and rospy.get_time() - self.last_distance_detected_time < self.stale_time) \
    and (self.last_rotation_detected_time and rospy.get_time() - self.last_rotation_detected_time < self.stale_time) \
    and self.lane_following:
      self.lane_follow("False")
      self.lane_following = False

    if self.last_range_detected_time and rospy.get_time() - self.last_range_detected_time < self.stale_time:
      self.range = None

  def drive(self):
    if self.lane_following:
      return
    
    # Determine Omega - based on lane-following
    if not self.distance_from_robot or not self.rotation_of_robot:
      self.twist.v = 0
    else:
      # Velocity control
      if (self.distance_from_robot < self.following_distance) \
      or (self.range and self.range < self.following_distance):
        self.twist.v = 0
      else:
        self.twist.v = self.velocity

      # Angle control
      # P Term
      angle_P = -self.angle_proportional * self.angle_P

      # D Term
      angle_d_error = (self.angle_proportional - self.last_angle_error) / (rospy.get_time() - self.last_angle_time)
      self.last_angle_error = self.angle_proportional
      self.last_angle_time = rospy.get_time()
      angle_D = angle_d_error * self.angle_D

      self.twist.omega = angle_P + angle_D

      # Publish command
      if DEBUG:
        print('[DEBUG]', self.distance_proportional, self.twist.omega, self.twist.v)
    self.vel_pub.publish(self.twist)

  def turn_on_light(self):
    '''
    Code for this function was inspired by 
    "duckietown/dt-core", file "led_emitter_node.py"
    Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
    Author: GitHub user liampaull
    '''
    self.pattern.header.stamp = rospy.Time.now()
    rgba_white = ColorRGBA()

    rgba_white.r = 1.0
    rgba_white.g = 1.0
    rgba_white.b = 1.0
    rgba_white.a = 1.0

    self.pattern.rgb_vals = [rgba_white] * 5
    
    self.color_publisher.publish(self.pattern)

  def hook(self):
    print("SHUTTING DOWN")
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)
    for i in range(8):
      self.vel_pub.publish(self.twist)

if __name__ == "__main__":
  node = DuckiebotFollowNode("duckiebot_follow_node")
  rate = rospy.Rate(8)  # 8hz
  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()