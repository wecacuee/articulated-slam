/* Defining all the function of camera pose estimation class */
#include <camera_pose.h>

/**
 * @brief Default constructor for Camera_Pose class
 */
Camera_Pose::Camera_Pose(void){
    // To check if previous cloud has been set -- used for calculating the differential motion
    initialized = false;
    // Initializing a boost shared pointer for storing the previous cloud
    cloudrgbptr dummy_ptr(new cloudrgb());
    cloud_prev = dummy_ptr;
    // Default method of evaluating the difference between clouds is dist_thresh
    diff_method = "dist_thresh"; // Available methods: "diff_octree", "dist_thresh"
    diff_thresh = 0.5;
}

/**
 * @brief Get the pose estimate of the current frame related to previous frame
 *
 * @param cloud The input cloud from the current frame
 */
void Camera_Pose::get_pose(const sensor_msgs::PointCloud2ConstPtr& cloud){
    // For removing NaN indices    
    std::vector <int> indices1,indices2;
    //ros::Time start_time = ros::Time::now();
    // If the previous cloud has not been set then the tranformation is I
    if (initialized){
        // To get the raw input cloud
        cloudrgbptr cloud_in(new cloudrgb());
        pcl::fromROSMsg(*cloud,*cloud_in);

        
        // Raw input cloud seems to have NaNs and InFs -- doesn't work with ICP
        cloudrgbptr cloud_in_icp(new cloudrgb());
        cloudrgbptr cloud_prev_icp(new cloudrgb());
        // removeNaNFromPointCloud is a terrible function because your point cloud is no more organized 
        pcl::removeNaNFromPointCloud(*cloud_in,*cloud_in_icp,indices1);
        pcl::removeNaNFromPointCloud(*cloud_prev,*cloud_prev_icp,indices2);

        // Get ICP
        pcl::IterativeClosestPoint<pcl::PointXYZRGB,pcl::PointXYZRGB> icp;
        icp.setInputSource(cloud_in_icp);
        icp.setInputTarget(cloud_prev_icp);
        // Setting other paramters of ICP
        icp.setMaximumIterations(3);
        icp.align(*cloud_prev_icp);
        // Check if ICP actually converged
        if (icp.hasConverged()){
            // Getting the pose estimate
            ROS_INFO_STREAM("Got Pose Estimate for current frame \n" 
                    <<std::setprecision(2)<<icp.getFinalTransformation());
            pose_estimates.push_back(icp.getFinalTransformation());

            // In case we are succesfully able to estimate transfor,
            // subtract the cloud after registering and see the change
            subtract_clouds(cloud_in);
        }
        else{
            ROS_INFO_STREAM("ICP didn't converge for current frame");
        }
        
        // Resetting the previous cloud to the current cloud input
        copyPointCloud(*cloud_in,*cloud_prev);
        //ros::Time end_time = ros::Time::now();
        //std::cout<<"Time difference measuresed is "<<end_time-start_time<<std::endl;
    }
    else{
        initialized = true;
        // Set the input cloud for the previous frame
        pcl::fromROSMsg(*cloud,*cloud_prev);

    }
}

/**
 * @brief Subtracts two point clouds once one knows the transformation between two frames
 * @param inp_cloud Current input cloud pointer
 */
void Camera_Pose::subtract_clouds(const cloudrgbptr inp_cloud){

    // Following the tutorial : http://www.pointclouds.org/documentation/tutorials/octree_change.php

    // Verifying that the clouds have same width and height
    assert(inp_cloud->width==cloud_prev->width);
    assert(inp_cloud->height==cloud_prev->height);

    // PCL point type to store the outlier data
    cloudrgbptr motion_outliers;
    motion_outliers.reset (new pcl::PointCloud<pcl::PointXYZRGB>(*inp_cloud) );

    if (diff_method.compare("diff_octree")==0){
        ROS_INFO_STREAM("Using diff octree to compare two point clouds");
        // To store index of points that are outliers
        std::vector<int> newPointIdxVector;
        // Octree resolution - side length of octree voxels
        float resolution = 32.0f;
        pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB> octree (resolution);
        // Add points from cloudA to octree
        octree.setInputCloud (inp_cloud);
        octree.addPointsFromInputCloud ();
        // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
        octree.switchBuffers ();
        // Add points from cloudB to octree
        octree.setInputCloud (cloud_prev);
        octree.addPointsFromInputCloud ();
        // Get vector of point indices from octree voxels which did not exist in previous buffer
        octree.getPointIndicesFromNewVoxels (newPointIdxVector);
        // Creating a new point cloud with difference
        // Following code from here for the DIFF_MODE
        //http://kfls2.googlecode.com/svn/trunk/apps/src/openni_change_viewer.cpp
        motion_outliers->points.reserve(newPointIdxVector.size());
        for (std::vector<int>::iterator it = newPointIdxVector.begin(); it != newPointIdxVector.end (); it++){
            motion_outliers->points[*it].rgb = 255<<16;
        }
        ROS_INFO_STREAM("New voxel count "<<newPointIdxVector.size());
    }
    else if (diff_method.compare("dist_thresh")==0){
        ROS_INFO_STREAM("Comparing two point clouds based on distance threshold");
        float z_diff;
        int count_pt = 0;
        // Following code from here http://docs.pointclouds.org/trunk/occlusion__reasoning_8h_source.html#l00166
        for (size_t i=0; i<inp_cloud->width;i++ ){
            for (size_t j=0; j<inp_cloud->height;j++ ){
                // Check for NaN values in both clouds
                if ((pcl::isFinite(inp_cloud->at(i,j)))&&(pcl::isFinite(cloud_prev->at(i,j)))){
                    // Check the difference in z value
                    z_diff = fabs(cloud_prev->at(i,j).z-inp_cloud->at(i,j).z);
                    if (z_diff>diff_thresh){
                        motion_outliers->at(i,j).rgb = 255<<16;
                        count_pt++;
                        //ROS_INFO_STREAM("x,y"<<inp_cloud->at(i,j).x<<","<<inp_cloud->at(i,j).y<<","<<z_diff);
                    }
                } 
            }
        }
        ROS_INFO_STREAM("Number of diff points is "<<count_pt);
    }
    else{
        ROS_INFO_STREAM("No valid point cloud comparision method found");
    }
    
    // Converting to ROS sensor_msgs::PointCloud2 for display
    sensor_msgs::PointCloud2 cloud_processed;

    //Convert the pcl cloud back to rosmsg
    pcl::toROSMsg(*motion_outliers, cloud_processed);
    //Set the header of the cloud
    cloud_processed.header.frame_id = cloud_prev->header.frame_id;
    // Publish the data
    //You may have to set the header frame id of the cloud_processed also
    display_motion.publish (cloud_processed);

}

