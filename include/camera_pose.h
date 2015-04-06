/* Defining a class object to measure the camera pose between two frames */

#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/octree/octree.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> cloudrgb;
typedef cloudrgb::Ptr cloudrgbptr;

class Camera_Pose
{
	public:
		Camera_Pose(); // Default constructor
        // Actual mechanism to get pose
		void get_pose(const sensor_msgs::PointCloud2ConstPtr& cloud); 
		void subtract_clouds(const cloudrgbptr inp_cloud); 
        std::vector<Eigen::Matrix4f> pose_estimates;
        ros::Publisher display_motion; // Show change between two frames
        ros::Publisher display_transformed; // Show change between two frames
        ros::Publisher display_prev; // To display the previous point cloud
        // Diff methods currently allowed are "dist_thresh","diff_octree"
        std::string diff_method; // Method to use for calculating difference
        float diff_thresh; // Threshold for distance between two points in a point cloud
	private:
		bool initialized; // To see if a previous cloud is set
		cloudrgbptr cloud_prev; // To store a previous cloud in organized form
        /* To store a previous cloud whose NaN have been removed and
         *  as a result is in unorganized form */
        cloudrgbptr cloud_prev_icp; 
};

#endif
