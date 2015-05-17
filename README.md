# Purpose
ROS package to segment articulated objects from a scene

#ToDO 2015May16

[X] Debug : Why non-postive semi definite matrix

[ ] Temporal models: Prismatic models can have zero velocity.

[ ] 2D to 3D. Model general 2D motion as a parameteric model. Prismatic and
revolute are subsets of it with some constraints. The rest of the 2D motion
space with complementary constraints is our general 2D motion.

[ ] 3D simulation/results

[ ] Select only one motion model deterministically
[ ] Mapping with noise

[ ] SLAM Localization and Mapping

# Usage

Make the python file get_data.py executable and run the code to get data

chmod +x get_data.py && ./get_data.py

For recorded bag use 

roslaunch aae segmentation_recorded.launch 

where fname is the full path of rosbag recorded file


For live sensor use:
Make sure to launch 

i) roscore on host computer 

ii) Switch on motor power for arm to move it to capture position

iii) roslaunch /home/youbot/work/thesis/code/robot_initialize.launch on youbot computer

roslaunch aae segmentation_sensor.launch

This launch file by default doesn't record a bag of data. For recording a bag pass the record parameter as true

roslaunch aae segmentation_sensor.launch record_data:=True

#Helpful Links
To record video from rviz display use glc-capture and glc-play

http://robot.wpi.edu/wiki/index.php/ROS:rviz-video


#Codes and Datasets
http://wiki.ros.org/articulation_tutorials
