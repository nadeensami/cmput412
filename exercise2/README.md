# Exercise 2: ROS Development and Kinematics

This repository contains implementation solutions for exercise 2. For information about the project, please read the report at: https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-2

## Structure

Most of the source code is in the `packages/mypackage/src` directory. Here is a brief description of each file:

- `led_server.py`: Implements a node that starts up an LED server. This service changes the color of the LED lights according to a string message containing the color.
- `my_camera_node.py`: Implements a node that subscribes to the Duckietown topic where the camera feed is continuously published to, then republishes that image to a custom topic.
- `my_encoder.py`: Implements the multi-state task from part 2. Contains a node that subscribes to wheel encoders, transforms that information using kinematics, and uses that to accomplish the tasks specified in part 2 by publishing velocities to a topic that commands the wheels using those velocities. It also sends messages to the LED service defined in `led_server.py` to change the color of the LEDs.
- `my_publisher.py`: A simple publisher node that publishes a message containing the robot's host name every second.
- `my_subscriber.py`: A simple subscriber node, counterpart to `my_publisher.py`, that subscribes to that node and echoes the data it receives.
- `my_test_client.py`: A simple client for the service defined by `led_server.py` that reads from the terminal and requests the server to turn the LEDs that colour.

The results from various odometry readings, as well as a simple program to plot them, is in the `packages/mypackage/odometry_readings` directory.

## Execution:

To execute this project, comment/uncomment the nodes you would like to launch in the `packages/mypackage/launch/multiple_nodes.py` launch file. Currently, it is set to start an LED service, as well as a node that implements the Multi-State task from part 2. To run the program, ensure that the variable `$BOT` store your robot's host name, and run the following commands:

```
dts devel build -f -H $BOT
dts devel run -H $BOT
```

## Credit:

This code is built from a template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/duckietown/template-ros).

Build on top of by Nadeen Mohamed and Celina Sheng.

Code was also borrowed (and properly cited in code) from the following sources.

- https://docs.duckietown.org/daffy/duckietown-robotics-development/out/dt_infrastructure.html
- https://docs.duckietown.org/daffy/duckietown-robotics-development/out/odometry_modeling.html
- https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/wheels_driver/src/wheels_driver_node.py
- http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29
- https://get-help.robotigniteacademy.com/t/how-to-stop-your-robot-when-ros-is-shutting-down/225
- https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
- https://github.com/duckietown/dt-core/blob/daffy/packages/deadreckoning/src/deadreckoning_node.py
