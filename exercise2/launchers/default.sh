#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# roscore &
# sleep 5
# dt-exec rosrun my_package my_publisher_node.py
# dt-exec rosrun my_package my_subscriber_node.py
# rossrv
# rossrv info
# ls -al /code/catkin_ws/build/my_package
# ls -Ral /code/catkin_ws
# rossrv info my_package/ColorService
roslaunch my_package multiple_nodes.launch veh:=$VEHICLE_NAME
# rosrun my_package led_server.py

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
