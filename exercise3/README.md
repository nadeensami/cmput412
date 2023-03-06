# Exercise 3: Computer Vision for Robotics

This repository contains implementation solutions for exercise 3. For information about the project, please read the report at:

[Nadeen Mohamed's site](https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-3) or [Celina Sheng's site](https://sites.google.com/ualberta.ca/csheng2-cmput-412/exercise-3)


## Structure

### Augmented Reality

There are several packages in this file. Most of the source code related to Apriltag detection and odometry computation is in the `packages/augmented_reality_apriltag/src` and `packages/deadreckoning/src` directories respectively. Here is a brief description of each file:

- `packages/augmented_reality/src/augmented_reality.py`: Implements a node that subscribes to the camera stream, obtains transformations for the stream based on the homography matrix obtained from extrinsic calibration, draws points on the camera based on the transformations and a map file, and publishes the superimposed image. The purpose of this file was to understand computer vision and augmented reality basics.

- `packages/augmented_reality_apriltag/src/augmented_reality_apriltag.py`: Implements a node that detects Apriltag images. The node calculates the transformations between the Apriltag and the robot's camera and sends the lateral and angular coordinate information to a rostopic, which is used by the DeadReckoningNode.

- `packages/deadreckoning/src/deadreckoning_node.py`: Implements a node that performs the robot's odometry. It listens to a rostopic for the Apriltag Node's tranformation information and update the location of the robot according to the detected Apriltags's ground truth.

- `packages/deadreckoning/src/deadreckoning_node.launch`: Contains nodes for static transformations of the ground truth Apriltag locations.

- `packages/lane_following/src/lane_following_node.py`: A node that utilizes OpenCV and camera information in order to perform autonomous lane following.

All other files are template files and can be disregarded.


## Execution:

To execute this project, comment/uncomment the packages you would like to launch in the `launchers/default.sh` script.

Currently, it is set to start the dead reckoning and apriltag detection nodes.

To run the program, ensure that the variable `$BOT` stores your robot's host name, and run the following commands:

```
dts devel build -f -H $BOT
dts devel run -H $BOT
```

## Credit:

This code is built from the 412 exercise 3 template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/wagonhelm/cmput412_exercise3).

Build on top of by Nadeen Mohamed and Celina Sheng.

Code was also borrowed (and cited in-code) from the following sources.

- https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
- https://docs.duckietown.org/daffy/duckietown-classical-robotics/out/cra_basic_augmented_reality_exercise.html
- https://github.com/duckietown/dt-core/blob/daffy/packages/apriltag/src/apriltag_detector_node.py
- https://github.com/Coral79/exA-3/blob/44adf94bad728507608086b91fbf5645fc22555f/packages/augmented_reality_basics/include/augmented_reality_basics/augmented_reality_basics.py
- https://github.com/duckietown-ethz/cra1-template/blob/master/packages/augmented_reality_apriltag/src/renderClass.py
- https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
- http://edu.gaitech.hk/turtlebot/line-follower.html
