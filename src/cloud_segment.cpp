#include<scene_segmenter.h>
#include<signal.h>

/* Creating a custom SIGINT call*/
void mySigintHandler(int sig){

  ROS_INFO("I got orders to shutdown now");
  ROS_INFO("To Do: move the arm to home position");
  ros::shutdown();

}

int main(int argc,char** argv)
{
// Initialize ROS
ros::init(argc,argv,"segmentation");
// Create an object of class Scene_Segmenter
Scene_Segmenter aae_segment;
aae_segment.initParams();
ros::spin(); // ROS : Take over control

return 0;
}


