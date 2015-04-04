/* Define a class object to segment the scene and report resulting objects
*/

#ifndef SCENE_SEGMENTER_H
#define SCENE_SEGMENTER_H

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <camera_pose.h>

class Scene_Segmenter
{
    public:
        /* To read the associated parameters from a launch file
        */
        void initParams();

        /* Perform actual filtering*/
        void voxel_filtering(const sensor_msgs::PointCloud2ConstPtr& cloud);

        /* To move the robot arm to certain pose */
        void move_arm(std::vector<float>& arm_position);

        sensor_msgs::PointCloud2 cloud_filtered; // To store the filtered cloud


    private:
        bool prev_cloud; // To see if we already have a cloud
        ros::NodeHandle node;
        ros::Publisher pub; // Publishes to the /camera/depth/voxelized topic
        ros::Subscriber sub; // Subscribers from /camera/depth_registered/points
        ros::Publisher robot_arm; // To control the robot arm by publishing on /arm_1/arm_controller/position_command
        bool arm_initialized; // To see if the arm has been initialized for capturing
        int frame_count;
        /* To get the camera pose estimates*/
        Camera_Pose sensor_pose;
};
#endif
