TO DO
 - 3D visualization (both robot trajectory and simulation map) - Robot POV and bird's eye view - Vikas (1)
 - Email Steffen - Vikas (2)
 - Massaging real data and GT association - Vikas (3)
 - Visualization of landmark predictions - Vikas (4)
 - Investigate why drifting slam states - Madan (1)
 - Baseline 2  - constant velocity model (prismatic for all ldmks) - Madan (2)
 - Simulation world - Entirely dynamic map generation and error metrics (keep robot static for ~30 frames) - Madan (3)
 - Paper - writing and cleanup - Suren (1)
 - Case - When static changes to prismatic/revolute observation likelihood - Suren (2)
 - Clustering based on parameters and visualization - Suren (3)
 
 

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

#to read
http://arxiv.org/pdf/1502.01659.pdf by Sudeep and Teller (MIT)
* Possible contributions
    - Moving camera/robot requires prediction and temporal modeling (SLAM
    type approach)
    - temporal modeling enables to duck a moving door/punch and avoid
    collisions.
    - RGBD based dense track ICP vs sparse feature tracking
    - Detection and fitting of the object after it has estimated the joint and
    is re-seen by the robot.


#Codes and Datasets
http://wiki.ros.org/articulation_tutorials

#Ideas and TODO's from CVPR

* Dataset for articulation estimation
* Monocular 2D SFM (Line/Point/Objects) + articulation : combined derivation
* Hierarchy everywhere
* clustering of direction of  incremental motion to get prismatic and revolute
joints. Better than ICP.
* Convex relaxations (Read and use it always)
* Difference between predicted and observed is important for which
localization is important. (John leanord)
* Temporal models: Preserve higher order coefficients (acceleration/jerk)
    instead of lower order (position/velocity). Because in case of spring (or
    human) motion velocity may be different but acceleration or jerk are
    constant and different for spring as compared to human. Compare with DTW.
    Also think in direction of estimating the weight/force required to open
    door but just observing someone else operating it ... just like the fabric
    paper.
* It is better to use change in direction of surface normals rather than
relative movements of points of time to get revolute joints.

#Paper ideas
* Compare with Ashutosh Saxena' work of priors on object trajectories
(learning from demonstration) like mug trajectories). Lookup temporal modeling
of their model. Perhaps they use DTW.
* Applications: Dynamic SLAM, Object SLAM
* Motion planning in dynamic environments: Merge or not merge, entering garage
with closing doors. 
