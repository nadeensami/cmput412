# Exercise 2: ROS Development and Kinematics

This repository contains implementation solutions for exercise 2. For information about the project, please read the report at: https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-1

## Structure

- `colordetector`: Contains code to recognize most of the colors from an image view
- `helloFromRobot`: Contains code that, when executed using the instructions below, prints "Hello from <hostname>", depending on the hostname of your robot

## Execution:

To run the hello from robot program, ensure that the variable `$BOT` store your robot's host name, and run the following commands:

```
cd helloFromRobot
dts devel build -f -H $BOT
dts devel run -H $BOT
```
