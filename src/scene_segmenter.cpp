#include<scene_segmenter.h>
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
#include <pcl_ros/point_cloud.h>
#include "brics_actuator/JointPositions.h"

typedef pcl::PointCloud<pcl::PointXYZRGB> cloudrgb;
typedef cloudrgb::Ptr cloudrgbptr;


void Scene_Segmenter::initParams(){
  // Initialized the publisher here
  pub = node.advertise<sensor_msgs::PointCloud2> ("/camera/depth/voxelized", 1);
  robot_arm = node.advertise<brics_actuator::JointPositions> ("/arm_1/arm_controller/position_command",1);
  // Created a subscriber here
  sub = node.subscribe("/camera/depth_registered/points", 1, &Scene_Segmenter::voxel_filtering, this);
  arm_initialized = false;

}

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
    std::cout << "Joint " << armJointPositions[i].joint_uri << " = " << armJointPositions[i].value << " " << armJointPositions[i].unit << std::endl;

  }
  command.positions = armJointPositions;
  robot_arm.publish(command);
  std::cout<<"Send position command to the arm"<<std::endl;
}

void Scene_Segmenter::voxel_filtering(const sensor_msgs::PointCloud2ConstPtr& cloud){

  if (arm_initialized) {
    cloudrgbptr PC (new cloudrgb());
    cloudrgbptr PC_filtered (new cloudrgb());
    pcl::fromROSMsg(*cloud, *PC); //Now you can process this PC using the pcl functions 
    sensor_msgs::PointCloud2 cloud_filtered;


    // Perform the actual filtering
    pcl::VoxelGrid<pcl::PointXYZRGB> sor ;
    sor.setInputCloud (PC);
    sor.setLeafSize (0.01, 0.01, 0.01);
    sor.filter (*PC_filtered);

    //Convert the pcl cloud back to rosmsg
    pcl::toROSMsg(*PC_filtered, cloud_filtered);
    //Set the header of the cloud
    cloud_filtered.header.frame_id = cloud->header.frame_id;
    // Publish the data
    //You may have to set the header frame id of the cloud_filtered also
    pub.publish (cloud_filtered);
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


