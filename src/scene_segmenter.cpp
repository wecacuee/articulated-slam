#include<scene_segmenter.h>
#include<camera_pose.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <boost/units/io.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/conversion.hpp>
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/plane_angle.hpp>
#include <boost/units/io.hpp>
#include <boost/units/systems/angle/degrees.hpp>
#include <boost/units/conversion.hpp>
#include "brics_actuator/JointPositions.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> cloudrgb;
typedef cloudrgb::Ptr cloudrgbptr;

/* Initialize parameters and creates nodes for advertising and subscribing*/
void Scene_Segmenter::initParams(){

    frame_count = 0;
    // Initialized all the publisher here
    pub = node.advertise<sensor_msgs::PointCloud2> ("/segmented_plane", 1);
    sensor_pose.display_motion = node.advertise<sensor_msgs::PointCloud2> ("/motion_outliers", 1);
    sensor_pose.display_transformed = node.advertise<sensor_msgs::PointCloud2> ("/motion_transformed", 1);
    sensor_pose.display_prev = node.advertise<sensor_msgs::PointCloud2> ("/display_prev", 1);
    robot_arm = node.advertise<brics_actuator::JointPositions> ("/arm_1/arm_controller/position_command",1);
    
    // Initialize all the subscriber here
    sub = node.subscribe("/camera/depth_registered/points", 1000, &Scene_Segmenter::voxel_filtering, this);
    // Getting the parameter from yaml file if the arm needs to be initialized
    bool arm_config; // This parameter is to decide whether the configuration files
    // requires one to initialize the arm
    node.getParam("/segment/arm_initialize",arm_config);
    if (arm_config){
        arm_initialized = false;
        ROS_INFO_STREAM("The arm will be moved to the viewing position");
    }
    else{
        arm_initialized = true;
        ROS_INFO_STREAM("The arm will not be moved");
    }


}

/* Move the robot arm to capturing position */
void Scene_Segmenter::move_arm(std::vector<float>& arm_position){
    brics_actuator::JointPositions command;
    std::vector <brics_actuator::JointValue> armJointPositions;
    armJointPositions.resize(arm_position.size()); 
    std::stringstream jointName;
    // ::io::base_unit_info <boost::units::si::angular_velocity>).name();
    for (int i = 0; i < arm_position.size(); i++) {
        jointName.str("");
        jointName << "arm_joint_" << (i + 1);
        armJointPositions[i].joint_uri = jointName.str();
        armJointPositions[i].value = arm_position[i];
        armJointPositions[i].unit = boost::units::to_string(boost::units::si::radians);
        std::cout << "Joint " << armJointPositions[i].joint_uri << " = " << armJointPositions[i].value
            << " " << armJointPositions[i].unit << std::endl;

    }
    command.positions = armJointPositions;
    robot_arm.publish(command);
    std::cout<<"Send position command to the arm"<<std::endl;
}

/*Dummy code for voxel filtering a point cloud*/
void Scene_Segmenter::voxel_filtering(const sensor_msgs::PointCloud2ConstPtr& cloud){

    if (arm_initialized) {
        frame_count++;
        if (frame_count%10==0){
        sensor_pose.get_pose(cloud);
        }
        /*
        cloudrgbptr PC (new cloudrgb());
        //cloudrgbptr cloud_filtered (new cloudrgb());
        pcl::fromROSMsg(*cloud, *PC); //Now you can process this PC using the pcl functions 
        // Creating a XYZ cloud for segmentation process
        pcl::PointCloud<pcl::PointXYZ>::Ptr PC_xyz (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());
        copyPointCloud(*PC,*PC_xyz);
        sensor_msgs::PointCloud2 cloud_processed;


        // Create the filtering object: downsample the dataset using a leaf size of 1cm
        pcl::VoxelGrid<pcl::PointXYZ> sor ;
        sor.setInputCloud (PC_xyz);
        sor.setLeafSize (0.01, 0.01, 0.01);
        sor.filter (*cloud_filtered);

        // Segmentation of cloud part here
        // Create the segmentation object for the planar model and set all the parameters
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations (100);
        seg.setDistanceThreshold (0.02);
        // Segment the largest part from the cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given input cloud." << std::endl;
        }
        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size ()
        //<< " data points." << std::endl;

        *cloud_filtered = *cloud_plane;

        //Convert the pcl cloud back to rosmsg
        pcl::toROSMsg(*cloud_filtered, cloud_processed);
        //Set the header of the cloud
        cloud_processed.header.frame_id = cloud->header.frame_id;
        // Publish the data
        //You may have to set the header frame id of the cloud_processed also
        pub.publish (cloud_processed);
        */
    }
    else{
        // Send the initialization position command to the arm
        float view_position[] = {2.9496, 1.1345, -2.5481, 3.3597, 2.9234};
        //float view_position[] = {0.02, 0.02,-0.02, 0.03, 0.12};
        std::vector<float>view_pos(&view_position[0],&view_position[0]+5);
        move_arm(view_pos);
        arm_initialized = true;
    }

}


