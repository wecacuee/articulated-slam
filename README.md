# Purpose
ROS package to segment articulated objects from a scene

#ToDO:
[] Do floor segmentation and try Euclidean clustering and visualization

# Usage
For recorded bag use 

roslaunch aae segmentation_recorded.launch fname:=/home/surenkum/work/thesis/data/ros_recorded/data_recorded.bag

where fname is the full path of rosbag recorded file


For live sensor use:
Make sure to launch 

i) roscore on host computer 

ii) Switch on motor power for arm to move it to capture position

iii) roslaunch /home/youbot/work/thesis/code/robot_initialize.launch on youbot computer

roslaunch aae segmentation_sensor.launch
