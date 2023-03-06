#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# LAUNCHING APP
# Uncomment for odometry and apriltag detection:
dt-exec roslaunch duckietown_demos deadreckoning.launch
dt-exec roslaunch augmented_reality_apriltag augmented_reality_apriltag.launch veh:=$VEHICLE_NAME

# Uncomment for lane following:
# dt-exec roslaunch lane_following lane_following.launch veh:=$VEHICLE_NAME

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
