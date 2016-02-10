'''
Performs the main goal of this project, being able to use articulation
inside SLAM, by first
- Getting few observations and estimating the articulated models
- Using the motion parameters of those models to propagate the landmarks and the robot
- Using observations to update the EKF state (consisting of robot pose and motion parameters)

- Code is similar to threeptmap.py
                # v1.0 Need to update mode to v2.0
                #round(ap2vec.dot(axisvec)/np.linalg.norm(axisvec),3) <= np.around(np.linalg.norm(axisvec),3)
 v1.0 Need to update mode to v2.0
                #slam_state = slam_state+np.dot(K_mat,(np.array([r,theta])-z_pred))
                #slam_state = slam_state+np.dot(K_mat,(np.array([r,theta])-z_pred))
'''
import sys
import os
import copy

import numpy as np
import cv2
import landmarkmap3d as landmarkmap
from landmarkmap3d import rodrigues
import estimate_mm as mm # To figure out the right motion model
import pdb
import utils_plot as up
import scipy.linalg
import matplotlib.pyplot as plt
import dataio
from extracttrajectories import feature_odom_gt_pose_iter_from_bag

from itertools import imap, izip, tee as itr_tee, islice
from collections import namedtuple, deque
import csv

TrackedLdmk = namedtuple('TrackedLdmk', ['ts', 'pt3D'])
models_names = ['Revolute','Prismatic','Static']

SIMULATEDDATA = False 
PLOTSIM = False 
#def threeptmap():
#    nframes = 100
#    map_conf = [# static
#                dict(ldmks=np.array([[0, 0]]).T,
#                     inittheta=0,
#                     initpos=[x, y],
#                     deltheta=0,
#                     delpos=[0,0]) 
#                for x,y in zip([10] * 10 + range(10, 191, 20) + [190]*10 +
#                               range(10, 191, 20),
#                               range(10, 191, 20) + [190]*10 + 
#                               range(10, 191, 20) + [10] * 10
#                              )
#               ] + [# prismatic
#                dict(ldmks=np.array([[10,10]]).T,
#                     inittheta=0,
#                     initpos=[120,10],
#                     deltheta=0,
#                     delpos=[5,0]),
#                # revolute
#                dict(ldmks=np.array([[0,20]]).T, # point wrt rigid body frame
#                     inittheta=np.pi,            # initial rotation
#                     initpos=[160,40],            # origin of rigid body
#                     deltheta=-10*np.pi/180,     # rotation per frame
#                     delpos=[0,0])               # translation per frame
#               ]
#
#    lmmap = landmarkmap.map_from_conf(map_conf, nframes)
#    lmvis = landmarkmap.LandmarksVisualizer([0,0], [200, 200], frame_period=-1,
#                                         scale=3)
#    robtraj = landmarkmap.robot_trajectory(np.array([[110, 90], [140,60],
#                                                     [120,50], [110, 90], 
#                                                     [140, 60]]),
#                                           5, np.pi/10)
#    # angle on both sides of robot dir
#    maxangle = 45*np.pi/180
#    # max distance in pixels
#    maxdist = 120
#
#    return nframes, lmmap, lmvis, robtraj, maxangle, maxdist

""" Generate Simulated map configuration data """
def threeptmap3d():
    nframes = 110
    scale = 30.
    map_conf=   [#static
                dict(ldmks=np.array([[0,0,0,]]).T / scale,
                initthetas=[0,0,0],
                initpos=np.array([x,y,z]) / scale,
                delthetas=[0,0,0],
                delpos=np.array([0,0,0]) / scale)
                for x,y,z in zip([10]*10 + range(10,191,20)+[190]*10+range(10,191,20),
                                 range(10,191,20)+[190]*10+range(10,191,20)+[10]*10,
                                 [5]*10 + range(1,11,1)+[1]*10+range(1,11,1))
                ]+[#Prismatic
                dict(ldmks=np.array([[40,160,0]]).T / scale,
                initthetas=[0,0,0],
                initpos=np.array([0,0,0]) / scale,
                delthetas=[0,0,0],
                delpos=np.array([1,0,0]) / scale)
                ]+[#Revolute
                dict(ldmks=np.array([[10,10,0]]).T / scale,
                initthetas=[0,0,0],
                initpos=np.array([130,130,0]) / scale,
                delthetas=[0,0,np.pi/5],
                delpos=np.array([0,0,0]) / scale)]
    
    lmmap = landmarkmap.map_from_conf(map_conf,nframes)
    # For now static robot 
    robtraj = landmarkmap.robot_trajectory(
        np.array([[110,90,0],[140,60,0],[120,50,0],[110,90,0],[140,60,0]]) / scale,
        0.2, np.pi/50, True, 100/scale, np.array([40, 40])/scale, nframes)
    maxangle = 45*np.pi/180
    maxdist = 120 / scale
    return nframes,lmmap,robtraj,maxangle,maxdist


""" Rotation matrix to quaternion conversion (including correction for gimbal lock problem)""" 
def Rtoquat(R):

    tr = R[0,0] + R[1,1] + R[2,2]
    if tr>0:
        S = np.sqrt(tr+1.0)*2
        qw = 0.25*S
        qx = (R[2,1] - R[1,2])/S
        qy = (R[0,2] - R[2,0])/S
        qz = (R[1,0] - R[0,1])/S

    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        qw = (R[2,1] - R[1,2])/S
        qx = 0.25*S
        qy = (R[0,1]+R[1,0])/S
        qz = (R[0,2]+R[2,0])/S

    elif R[1,1]>R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        qw = (R[0,2]-R[2,0])/S
        qx = (R[0,1]+R[1,0])/S
        qy = 0.25*S
        qz = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        qw = (R[1,0]-R[0,1])/S
        qx = (R[0,2]+R[2,0])/S
        qy = (R[1,2]+R[2,1])/S
        qz = 0.25*S

    return (qx,qy,qz,qw)

#def visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state_2D,
#                              slam_cov_2D, colors,obs_num):
#    thisframe = lmvis.genframe(ldmks, robview, colors)
#    thisframe = lmvis.drawrobot(robview, thisframe)
#    theta, width, height = up.ellipse_parameters_from_cov(slam_cov_2D,
#                                                          volume=0.50)
#    cv2.ellipse(thisframe, 
#                tuple(map(np.int32, slam_state_2D * lmvis._scale)),
#                tuple(np.int32(x * lmvis._scale) for x in (width/2, height/2)),
#                theta, 0, 360,
#                (0,0,255),2)
#    cv2.imshow(lmvis._name, thisframe)
#    filename = '../media/ekf_frame%04d.png' % obs_num 
#    print 'Writing frame to %s' % filename
#    cv2.imwrite(filename,thisframe)
#    cv2.waitKey(lmvis.frame_period)

'''
Propagates robot motion with two different models, one for linear velocity
and other one for a combination of linear and rotational velocity
Inputs are: 
Previous robot state,
covarinace in previous state,
actual robot input (translational and rotational component),
and time interval
'''
def robot_motion_prop(prev_state,prev_state_cov,robot_input,delta_t=1, 
                     motion_model='nonholonomic'):
    # Robot input is [v,w]^T where v is the linear velocity and w is the rotational component
    v = robot_input[0];w=robot_input[1];
    # Robot state is [x,y,\theta]^T, where x,y is the position and \theta is the orientation
    x = prev_state[0]; y=prev_state[1];theta = prev_state[2]
    robot_state = np.zeros(3)
    # Setting noise parameters, following Page 210 Chapter 7, Mobile Robot Localization of 
    alpha_1 = 0.1; alpha_2=0.05; alpha_3 = 0.05; alpha_4 = 0.1
    
    if motion_model == 'holonomic':
        vx, vy = v
        robot_state = np.array([
            x+(vx*np.cos(theta) - vy*np.sin(theta)) * delta_t,
            y+(vx*np.sin(theta) + vy*np.cos(theta)) * delta_t,
            theta + w * delta_t
        ])
        M = np.array([[(alpha_1*(vx**2))+(alpha_1*(vy**2))+(alpha_2*(w**2)),0, 0],
                      [0, (alpha_1*(vx**2))+(alpha_1*(vy**2))+(alpha_2*(w**2)), 0],
                      [0, 0, (alpha_3*(vx**2))+(alpha_3*(vy**2))+(alpha_4*(w**2))]])
        # G = \del robot_state / \del [x,y,theta]
        G = np.array(
            [[1, 0, (-vx * np.sin(theta) - vy * np.cos(theta)) * delta_t],
             [0, 1, (vx * np.cos(theta) - vy * np.sin(theta) ) * delta_t],
             [0, 0,                                                    1]])

        # V = \del robot_state / \del [vx, vy, w]
        V = np.array(
            [[np.cos(theta) * delta_t, -np.sin(theta) * delta_t, 0],
             [np.sin(theta) * delta_t,  np.cos(theta) * delta_t, 0],
             [                      0,                        0, 1]])
    elif motion_model == 'nonholonomic':
        # M for transferring noise from input space to state-space
        M = np.array([[(alpha_1*(v**2))+(alpha_2*(w**2)),0],[0,(alpha_3*(v**2))+(alpha_4*(w**2))]])
        # Check if rotational velocity is close to 0
        if (abs(w)<1e-4):
            robot_state[0] = x+(v*delta_t*np.cos(theta))        
            robot_state[1] = y+(v*delta_t*np.sin(theta))        
            robot_state[2] = theta
            # Derivative of forward dynamics model w.r.t previous robot state
            G = np.array([[1,0,-v*delta_t*np.sin(theta)],\
                    [0,1,v*delta_t*np.cos(theta)],\
                    [0,0,1]])
            # Derivative of forward dynamics model w.r.t robot input
            V = np.array([[delta_t*np.cos(theta),0],[delta_t*np.sin(theta),0],[0,0]])
        else:
            # We have a non-zero rotation component
            robot_state[0] = x+(((-v/w)*np.sin(theta))+((v/w)*np.sin(theta+w*delta_t)))
            robot_state[1] = y+(((v/w)*np.cos(theta))-((v/w)*np.cos(theta+w*delta_t)))
            robot_state[2] = theta+(w*delta_t)
            G = np.array([[1,0,(v/w)*(-np.cos(theta)+np.cos(theta+w*delta_t))],\
                    [0,1,(v/w)*(-np.sin(theta)+np.sin(theta+w*delta_t))],\
                    [0,0,1]])
            # Derivative of forward dynamics model w.r.t robot input
            # Page 206, Eq 7.11
            V = np.array([[(-np.sin(theta)+np.sin(theta+w*delta_t))/w,\
                    (v*(np.sin(theta)-np.sin(theta+w*delta_t)))/(w**2)+((v*np.cos(theta+w*delta_t)*delta_t)/w)],\
                    [(np.cos(theta)-np.cos(theta+w*delta_t))/w,\
                    (-v*(np.cos(theta)-np.cos(theta+w*delta_t)))/(w**2)+((v*np.sin(theta+w*delta_t)*delta_t)/w)],\
                    [0,delta_t]])
    else:
        raise ValueError('Unknown motion model %s' % motion_model)
    # Covariance in propagated state
    state_cov = np.dot(np.dot(G,prev_state_cov),np.transpose(G))+np.dot(np.dot(V,M),np.transpose(V))
    return robot_state,state_cov

class FilterTrackByLength(object):
    def __init__(self):
        self._track_lengths = dict()
        self._min_track_length = 200

    def count(self, rob_obs_iter):
        for timestamp, ids,rob_state_and_input, ldmks, ldmk_robot_obs in rob_obs_iter:
            for id in ids:
                self._track_lengths[id] = self._track_lengths.get(id, 0) + 1

    def isvalid(self, id):
        return self._track_lengths.get(id, 0) > self._min_track_length

    def filter(self, rob_obs_iter):
        for timestamp, ids,rob_state_and_input, ldmks, ldmk_robot_obs in rob_obs_iter:
            f_ids = [id for id in ids if self.isvalid(id)]
            bool_valid = np.array([(True if self.isvalid(id) else False)
                                   for id in ids])

            f_ldmk_robot_obs = ldmk_robot_obs[:, bool_valid]
            yield (timestamp, f_ids, rob_state_and_input,
                   ldmks, f_ldmk_robot_obs)

class RealDataToSimulatedFormat(object):
    def __init__(self, K, max_depth=4):
        # (z_view) real_k assumes Z in viewing direction, Y-axis downwards
        # (x_viwe) Simulated data format assumes X in viewing direction, Z -axis upwards
        R_x_view_2_z_view = rodrigues([0, 1, 0], -np.pi/2).dot(
            rodrigues([1, 0, 0], np.pi/2))
        K_x_view = K.dot(R_x_view_2_z_view)
        self.camera_K_x_view = K_x_view
        self._first_gt_pose = None
        self._all_ldmks = np.zeros((2,0))
        self._max_depth = max_depth

    def first_gt_pose(self):
        return self._first_gt_pose

    def compute_ldmks_robot_pose(self, points, depths):
        points_a = np.array(points).T
        depths_a = np.array(depths)
        Kinv = np.linalg.inv(self.camera_K_x_view)
        ldmk_robot_obs = Kinv.dot(landmarkmap.euc2homo(points_a)) * depths_a
        return ldmk_robot_obs

    def prepare_ldmks_all_world(self, theta, ldmk_robot_obs, xy, track_ids):
        # prepare ldmks
        R_c2w = rodrigues([0, 0, 1], theta)
        ldmks_world = R_c2w.dot(ldmk_robot_obs) \
                + np.asarray([xy[0], xy[1], 0]).reshape(-1, 1)
        max_track_id = np.maximum(np.max(track_ids), self._all_ldmks.shape[1])
        all_ldmks = np.zeros((ldmks_world.shape[0], max_track_id + 1))
        if (self._all_ldmks.shape[1]):
            all_ldmks[:, :self._all_ldmks.shape[1]] = self._all_ldmks
        all_ldmks[:, track_ids] = ldmks_world
        self._all_ldmks = all_ldmks
        return self._all_ldmks

    def unpack_odom(self, odom):
        (pos, quat), (linvel, angvel) = odom
        assert(abs(linvel[2]) < 0.1)
        assert(abs(angvel[0]) < 0.1)
        assert(abs(angvel[1]) < 0.1)
        return linvel[:2], angvel[2]

    def unpack_gt_pose(self, gt_pose):
        (xyz, abg) = gt_pose
        if self._first_gt_pose is None:
            self._first_gt_pose = copy.deepcopy(gt_pose)

        (_, _, z0), _ = self._first_gt_pose
        xyz[2] = xyz[2] - z0
        assert abs(xyz[2]) < 0.15, 'xyz[2] - z0 = %f; z0 = %f' % (xyz[2], z0)
        assert abs(abg[0]) < 0.05, 'abg[0] = %f' % abg[0]
        assert abs(abg[1]) < 0.05, 'abg[1] = %f' % abg[1]
        return (xyz[:2], abg[2])

    def real_data_to_required_format(self, ts_features_odom_gt):
        ts, features_odom_gt = ts_features_odom_gt
        tracked_pts, odom, gt_pose = features_odom_gt
        filtered_tracked_pts = [(id, pt, d) for id, pt, d in tracked_pts 
                                if d < self._max_depth]
        track_ids, points, depths  = zip(*filtered_tracked_pts)
        (linvel2D, angvel2D) = self.unpack_odom(odom)
        (xy, theta) = self.unpack_gt_pose(gt_pose)
        ldmk_robot_obs = self.compute_ldmks_robot_pose(points, depths)
        all_ldmks = self.prepare_ldmks_all_world(theta, ldmk_robot_obs,
                                     xy, track_ids)

        return (ts, track_ids, [xy[0], xy[1], theta,
                            np.asarray(linvel2D), angvel2D],
                all_ldmks, ldmk_robot_obs)

def get_timeseries_data_iter(timeseries_data_file):
    serializer = dataio.FeaturesOdomGTSerializer()
    datafile = open(timeseries_data_file)
    timestamps = serializer.init_load(datafile)
    data_iter = izip(timestamps, serializer.load_iter(datafile))
    return data_iter
    

def plot_sim_res(PLOTSIM,prob_plot1,prob_plot2,prob_plot3,traj_ldmk1,traj_ldmk2,traj_ldmk3,true_robot_states,slam_robot_states):
    prob_plot1_dup = prob_plot1
    prob_plot2_dup = prob_plot2
    prob_plot3_dup = prob_plot3
    prob_plot1 = np.dstack(prob_plot1)[0]
    prob_plot2 = np.dstack(prob_plot2)[0]
    prob_plot3 = np.dstack(prob_plot3)[0]

    plt.figure('Prismatic')
    h1 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[0],'o-b',label='Revolute',linewidth=3.0,markersize=15.0)
    h2 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[1],'*-g',label='Prismatic',linewidth=3.0,markersize=15.0)
    h3 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[2],'^-r',label='Static',linewidth = 3.0,markersize=15.0)
    plt.xlabel('Number of frames',fontsize=24)
    plt.ylabel('Probability',fontsize=24)
    plt.xticks([0,2,4,6,8,10,12,14,16,18],fontsize=24)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=24)
    plt.legend(loc=3,fontsize=24)

    plt.figure("Revolute")
    h1= plt.plot(range(len(prob_plot2_dup)),prob_plot2[0],'o-b',label='Revolute',linewidth = 3.0,markersize=15.0)
    h2 = plt.plot(range(len(prob_plot2_dup)),prob_plot2[1],'*-g',label='Prismatic',linewidth = 3.0,markersize=15.0)
    h3 = plt.plot(range(len(prob_plot2_dup)),prob_plot2[2],'^-r',label='Static',linewidth = 3.0,markersize=15.0)
    plt.xlabel('Number of frames',fontsize=24)
    plt.ylabel('Probability',fontsize=24)
    plt.xticks([0,2,4,6,8,10,12,14,16,18,19],fontsize=14)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=24)
    plt.legend(loc=3,fontsize=24)

    plt.figure("Static")
    h1= plt.plot(range(len(prob_plot3_dup)),prob_plot3[0],'o-b',label='Revolute',linewidth=3.0,markersize=15.0)
    h2 = plt.plot(range(len(prob_plot3_dup)),prob_plot3[1],'*-g',label='Prismatic',linewidth=3.0,markersize=15.0)
    h3 = plt.plot(range(len(prob_plot3_dup)),prob_plot3[2],'^-r',label='Static',linewidth=3.0,markersize=15.0)
    plt.xlabel('Number of frames',fontsize=24)
    plt.ylabel('Probability',fontsize=24)
    plt.xticks([0,2,4,6,8],fontsize=24)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=24)
    plt.legend(loc=3,fontsize=24)

    plt.figure('Trajectories')
    true_robot_states = np.dstack(true_robot_states)[0]
    slam_robot_states = np.dstack(slam_robot_states)[0]
    traj_ldmk1 = np.dstack(traj_ldmk1)[0]
    traj_ldmk2 = np.dstack(traj_ldmk2)[0]
    traj_ldmk3 = np.dstack(traj_ldmk3)[0]
    plt.plot(true_robot_states[0],true_robot_states[1],'-k',linestyle='dashed',label='True Robot trajectory',markersize=15.0)
    plt.plot(slam_robot_states[0],slam_robot_states[1],'^g',label='A-SLAM trajectory',markersize=15.0)

    plt.plot(traj_ldmk1[0],traj_ldmk1[1],'*-g',linestyle='dotted',label='Prismatic joint',markersize=15.0)
    plt.plot(traj_ldmk2[0],traj_ldmk2[1],'o-b',linestyle='dotted',label='Revolute joint',markersize=10.0)
    plt.plot(traj_ldmk3[0],traj_ldmk3[1],'^-r',label='Static joint',markersize=15.0)
    plt.xticks([-2,0,2,4,6],fontsize=24)
    plt.yticks([-2,0,2,4,6],fontsize=24)
    plt.legend(loc=4,fontsize=24)
    plt.show()



'''
Performing Articulated SLAM
Pass in optional parameter for collecting debug output for all the landmarks
'''
def articulated_slam(debug_inp=True):
    # Writing to file variables
    f_gt = open('gt.txt','w')
    f_slam = open('slam.txt','w')    
    img_shape = (240, 320)
    f = 300
    camera_K_z_view = np.array([[f, 0, img_shape[1]/2.], 
                       [0, f, img_shape[0]/2.],
                       [0,   0,   1]])

    # Motion probability threshold
    m_thresh = 0.60 # Choose the articulation model with greater than this threhsold's probability
    
    # Getting the map
    nframes,lmmap,robtraj,maxangle,maxdist = threeptmap3d()
    #nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ekf_map_id = dict(); # Formulating consistent EKF mean and covariance vectors
    
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    
    # Handle simulated and real data , setup required variables
    if SIMULATEDDATA:
        rob_obs_iter = landmarkmap.get_robot_observations(lmmap, robtraj,
                                                          maxangle, maxdist,
                                                          img_shape,
                                                          camera_K_z_view,
                                                  # Do not pass visualizer to
                                                  # disable visualization
                                                  lmvis=None)
        motion_model = 'nonholonomic'
        robview = landmarkmap.RobotView(img_shape, camera_K_z_view, maxdist)
        csv = np.genfromtxt('expt_noise.csv',delimiter=',')
        count = 0 

    else:

        timeseries_data_file = "../mid/articulatedslam/2016-01-22/rev_2016-01-22-13-56-28/extracttrajectories_GFTT_SIFT_odom_gt_timeseries.txt"
        bag_file = "../mid/articulatedslam/2016-01-22/rev2_2016-01-22-14-32-13.bag"
        timeseries_dir = os.path.dirname(timeseries_data_file)
        image_file_fmt = os.path.join(timeseries_dir, "img/frame%04d.png")
        timestamps_file = os.path.join(timeseries_dir, 'img/timestamps.txt')

        robview = landmarkmap.RobotView(img_shape, camera_K_z_view, maxdist,
                                   image_file_format=image_file_fmt,
                                   timestamps_file=timestamps_file)
        use_bag = False 
        if use_bag:
            data_iter = feature_odom_gt_pose_iter_from_bag(bag_file, "GFTT",
                                                       "SIFT")
        else:
            data_iter = get_timeseries_data_iter(timeseries_data_file)

        VISDEBUG = 0
        if VISDEBUG:
            plotables = dict()
            for i, (ts, (features, odom, gt_pose)) in enumerate(data_iter):
                if i % 100 != 0:
                    continue
                (pos, quat), (linvel, angvel) = odom
                xyz, abg = gt_pose
                plotables.setdefault('linvel[0]', []).append(linvel[0])
                plotables.setdefault('linvel[1]', []).append(linvel[1])
                plotables.setdefault('linvel[2]', []).append(linvel[2])
                plotables.setdefault('angvel[0]', []).append(angvel[0])
                plotables.setdefault('angvel[1]', []).append(angvel[1])
                plotables.setdefault('angvel[2]', []).append(angvel[2])
                plotables.setdefault('xyz[0]', []).append(xyz[0])
                plotables.setdefault('xyz[1]', []).append(xyz[1])
                plotables.setdefault('xyz[2]', []).append(xyz[2])
                plotables.setdefault('abg[0]', []).append(abg[0])
                plotables.setdefault('abg[1]', []).append(abg[1])
                plotables.setdefault('abg[2]', []).append(abg[2])

            for (k, v), m in zip(plotables.items(), ',.1234_x|^vo'):
                plt.plot(v, label=k, marker=m)
            plt.legend(loc='best')
            plt.show()
            sys.exit(0)

        formatter = RealDataToSimulatedFormat(camera_K_z_view)
        rob_obs_iter_all = imap(
            formatter.real_data_to_required_format,
            data_iter)
        filter_by_length = FilterTrackByLength()
        filter_by_length.count(islice(rob_obs_iter_all, 500))

        if use_bag:
            data_iter = feature_odom_gt_pose_iter_from_bag(bag_file, "GFTT",
                                                       "SIFT")
        else:
            data_iter = get_timeseries_data_iter(timeseries_data_file)

        formatter = RealDataToSimulatedFormat(camera_K_z_view)
        rob_obs_iter_all = imap(
            formatter.real_data_to_required_format,
            data_iter)
        rob_obs_iter = filter_by_length.filter(islice(rob_obs_iter_all, 500))

        motion_model = 'holonomic'




    lmvis = landmarkmap.LandmarksVisualizer([0,0], [7, 7], frame_period=10,
                                            imgshape=(700, 700))
    
    # EKF parameters for filtering
    first_traj_pt = dict()
    # Initially we only have the robot state
    (_, _, rob_state_and_input, _, _) = rob_obs_iter.next()
    model = dict()
    slam_state =  np.array(rob_state_and_input[:3]) # \mu_{t} state at current time step
    
    # Covariance following discussion on page 317
    # Assuming that position of the robot is exactly known
    slam_cov = np.diag(np.ones(slam_state.shape[0])) # covariance at current time step
    ld_ids = [] # Landmark ids which will be used for EKF motion propagation step
    index_set = [slam_state.shape[0]] # To update only the appropriate indices of state and covariance 
    # Observation noise
    Q_obs = np.array([[5.0,0,0],[0,5.0,0],[0,0,5.0]])
    #For plotting
    obs_num = 0
    # For error estimation in robot localization
    true_robot_states = []
    slam_robot_states = []
    ldmktracks = dict()


    # Plotting variables for simulated data
    if PLOTSIM:
        prob_plot1 = []
        prob_plot2 = []
        prob_plot3 = []
        traj_ldmk1 = []
        traj_ldmk2 = []
        traj_ldmk3 = []    

    # Processing all the observations
    for fidx,(timestamp, ids,rob_state_and_input, ldmks, ldmk_robot_obs) in enumerate(rob_obs_iter):    
        if not SIMULATEDDATA:
            if fidx%5!=0:# or fidx < 200:
                continue
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state and inputs:', rob_state, robot_input
        print 'Observations size:',ldmk_robot_obs.shape
        
        # Following EKF steps now

        # First step is propagate : both robot state and motion parameter of any active landmark
        slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],
                                                            slam_cov[0:3,0:3],
                                                            robot_input,
                                                           motion_model=motion_model)
        # Active here means the landmark for which an articulation model has been associated
        if len(ld_ids)>0:
            '''
            Step 2: When any landmark's motion model is estimated, we start doing EKF SLAM with 
            state as robot's pose and motion parameter of each of the landmarks currently estimated
            Here we propagate motion parameters of each model
            '''
            for (ld_id,start_ind,end_ind) in zip(ld_ids,index_set[:-1],index_set[1:]):
                # Landmark with ld_id to propagate itself from the last state
                slam_state[start_ind:end_ind] = ldmk_am[ld_id].predict_motion_pars(\
                        slam_state[start_ind:end_ind])
                # Propagateontact directly to setup meeting

                slam_cov[start_ind:end_ind,start_ind:end_ind] = ldmk_am[ld_id].prop_motion_par_cov(slam_cov[start_ind:end_ind,start_ind:end_ind])
            # end of loop over ekf propagation

        # end of if



        colors = []
        mm_probs = []
        # Collecting all the predictions made by the landmark
        ld_preds = []
        ld_ids_preds = []
        ids_list = []
        lk_pred = None 
    
        # Processing all the observations
        # v2.0
        for id, ldmk_rob_obv in zip(ids,np.dstack(ldmk_robot_obs)[0]):
            # Adding noise
            if SIMULATEDDATA:
                ldmk_rob_obv = ldmk_rob_obv + csv[count,:]         
                count = count + 1
            
            if id not in first_traj_pt:
                
                #first_traj_pt[id]=ldmk_rob_obv
                Rc2w = rodrigues([0,0,1],slam_state[2])
                first_traj_pt[id] = Rc2w.T.dot(ldmk_rob_obv)+np.asarray([slam_state[0],slam_state[1],0.0])

 
            motion_class = ldmk_estimater.setdefault(id, mm.Estimate_Mm())
            # For storing the chosen articulated model
            chosen_am = ldmk_am.setdefault(id,None)
            '''
            Step 1: Process Observations to first determine the motion model of each landmark
            During this step we will use robot's internal odometry to get the best position of 
            external landmark as we can and try to get enough data to estimate landmark's motion
            model. 
            '''
            # For each landmark id, we want to check if the motion model has been estimated
            if ldmk_am[id] is None:
                motion_class.process_inp_data([0,0], slam_state[:3],ldmk_rob_obv,first_traj_pt[id])
                # Still need to estimate the motion class
                # Check if the model is estimated
                if sum(motion_class.prior>m_thresh)>0:
                    model[id] = np.where(motion_class.prior>m_thresh)[0]	
                    print("INFO: Model estimated %s " % models_names[int(model[id])])
                    ldmk_am[id] = motion_class.am[int(model[id])]
                    ld_ids.append(id)
                    curr_ld_state = ldmk_am[id].current_state()
                    curr_ld_cov = ldmk_am[id].current_cov()
                    index_set.append(index_set[-1]+curr_ld_state.shape[0])
                    # Extend Robot state by adding the motion parameter of the landmark
                    slam_state = np.append(slam_state,curr_ld_state)
                    # Extend Robot covariance by adding the uncertainity corresponding to the 
                    # robot state
                    slam_cov = scipy.linalg.block_diag(slam_cov,curr_ld_cov) 
            else:
                # This means this landmark is an actual observation that must be used for filtering
                # the robot state as well as the motion parameter associated with the observed landmark
                # Getting the predicted observation from the landmark articulated motion
                # Getting the motion parameters associated with this landmark
                curr_ind = ld_ids.index(id)
                # Following steps from Table 10.2 from book Probabilistic Robotics
                lk_pred = ldmk_am[id].predict_model(motion_class.means[int(model[id])])
                ld_preds.append(lk_pred)
                ld_ids_preds.append(curr_ind)

		        # v2.0
                R_temp = rodrigues([0,0,1],slam_state[2]).T
                pos_list = np.ndarray.tolist(slam_state[0:2])
                pos_list.append(0.0)
                # To match R_w2c matrix
                z_pred = R_temp.T.dot(lk_pred-np.array(pos_list)) 

                H_mat = np.zeros((3,index_set[-1]))
                curr_obs = ldmk_rob_obv
                theta = slam_state[2]
                H_mat[0,0:3] = np.array([-np.cos(theta),-np.sin(theta),-np.sin(theta)*curr_obs[0]+ np.cos(theta)*curr_obs[1]])
                H_mat[1,0:3] = np.array([np.sin(theta),-np.cos(theta),(-np.cos(theta)*curr_obs[0])-(np.sin(theta)*curr_obs[1])])
                H_mat[2,0:3] = np.array([0,0,0])

                H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = np.dot(R_temp,ldmk_am[id].observation_jac(slam_state[index_set[curr_ind]:index_set[curr_ind+1]],first_traj_pt[id]))


                # Innovation covariance
                inno_cov = np.dot(H_mat,np.dot(slam_cov,H_mat.T))+Q_obs
                # Kalman Gain
                K_mat = np.dot(np.dot(slam_cov,H_mat.T),np.linalg.inv(inno_cov))
                
                # Updating SLAM state
                # v2.0
                slam_state = slam_state + np.hstack((np.dot(K_mat,(np.vstack((ldmk_rob_obv-z_pred))))))               
                # Updating SLAM covariance
                slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)
            # end of if else ldmk_am[id]
            
            if PLOTSIM:
                if id == 40:
                    prob_plot1.append(motion_class.prior.copy())
                if id == 41:
                    prob_plot2.append(motion_class.prior.copy())
                if id == 13:
                    prob_plot3.append(motion_class.prior.copy())

            mm_probs.append(motion_class.prior)

            motion_class = ldmk_estimater[id]
            p1, p2, p3 = motion_class.prior[:3]
            color = np.int64((p1*rev_color
                     + p2*pris_color
                     + p3*stat_color))
            color = color - np.min(color)
            colors.append(color)
            ids_list.append(id)

        # end of loop over observations in single frame
        
        if PLOTSIM:
            if 40 in ldmk_am.keys() and ldmk_am[40] is not None:
                lk_pred = ldmk_am[40].predict_model(slam_state[index_set[ld_ids.index(40)]:index_set[ld_ids.index(40)]+1])
                traj_ldmk1.append(lk_pred.copy())
            if 41 in ldmk_am.keys() and ldmk_am[41] is not None:
                lk_pred = ldmk_am[41].predict_model(slam_state[index_set[ld_ids.index(41)]:index_set[ld_ids.index(41)]+1])
                traj_ldmk2.append(lk_pred.copy())
            if 13 in ldmk_am.keys() and ldmk_am[13] is not None:
                lk_pred = ldmk_am[13].predict_model(slam_state[index_set[ld_ids.index(13)]:index_set[ld_ids.index(13)]+1])
                traj_ldmk3.append(lk_pred.copy())

        
        # Follow all the steps on
        #print "SLAM State for robot and landmarks is",slam_state
        #obs_num = obs_num+1
        #print 'motion_class.priors', mm_probs
        #print 'ids:', ids_list
        #print 'colors:', colors
        ##up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        #visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
        #                          slam_cov[:2, :2], colors,obs_num)

        for i, id in enumerate(ids):
            ldmktracks.setdefault(id, []).append(
                TrackedLdmk(timestamp, ldmk_robot_obs[:, i:i+1]))

        # Handle new robot view code appropriately
        robview.set_robot_pos_theta((rob_state[0], rob_state[1], 0),
                                    rob_state[2]) 
        assert ldmk_robot_obs.shape[1] == len(colors), '%d <=> %d' % (
            ldmk_robot_obs.shape[1], len(colors))
        
        #Img = lmvis.genframe(ldmks, ldmk_robot_obs=ldmk_robot_obs, robview = robview,colors=colors,SIMULATEDDATA=SIMULATEDDATA)
        #Imgr = lmvis.drawrobot(robview, img)
        #Imgrv = robview.drawtracks([ldmktracks[id] for id in ids],
        #                           imgidx=robview.imgidx_by_timestamp(timestamp),
        #                           colors=colors)


        #if not SIMULATEDDATA:

        #    for id in ld_ids:
        #        if model[id] == 0:
        #            config_pars = ldmk_am[id].config_pars
        #            vec1 = config_pars['vec1']
        #            vec2 = config_pars['vec2']
        #            center = config_pars['center']
        #            center3D = center[0]*vec1 + center[1]*vec2 + ldmk_am[id].plane_point
        #            axis_vec = np.cross(vec1, vec2)
        #            radius = config_pars['radius']
        #            imgrv = robview.drawrevaxis(imgrv, center3D, axis_vec, radius, rev_color)
        #            imgr = lmvis.drawrevaxis(imgr, center3D, axis_vec, radius,
        #                                     rev_color)

        #robview.visualize(imgrv)
        #lmvis.imshow_and_wait(imgr)
            

        quat_true = Rtoquat(rodrigues([0,0,1],rob_state[2]))
        quat_slam = Rtoquat(rodrigues([0,0,1],slam_state[2]))    
        if fidx < 1000:
            f_gt.write(str(fidx+1)+" "+str(rob_state[0])+" "+str(rob_state[1])+" "+str(0)+" "+str(quat_true[0])+" "+str(quat_true[1])+" "+str(quat_true[2])+" "+str(quat_true[3])+" "+"\n")
            f_slam.write(str(fidx+1)+" "+str(slam_state[0])+" "+str(slam_state[1])+" "+str(0)+" "+str(quat_slam[0])+" "+str(quat_slam[1])+" "+str(quat_slam[2])+" "+str(quat_slam[3])+" "+"\n")
            true_robot_states.append(rob_state)
            slam_robot_states.append(slam_state[0:3].tolist())

        obs_num = obs_num + 1
        print 'SLAM state:',slam_state[0:4]
    # end of loop over frames
    
    if PLOTSIM:
    # Plot experimental results 
        plot_sim_res(PLOTSIM,prob_plot1,prob_plot2,prob_plot3,traj_ldmk1,traj_ldmk2,traj_ldmk3,true_robot_states,slam_robot_states)

    # Debugging
    if debug_inp is True:
        # Going over all the landmarks
        for ldmk_id,ldmk in ldmk_estimater.iteritems():
            if ldmk.num_data>ldmk.min_samples:
                # This landmark has atleast got enought observations for estimating motion parameters
                if ldmk_am[ldmk_id] is None:
                    print "Could not estimate model for landmark ", ldmk_id,\
                            "with ", ldmk.num_data, " samples"
    f_gt.close()
    f_slam.close()
    return (true_robot_states,slam_robot_states)


if __name__ == '__main__':
    # For reproducing the similar results with newer version of code
    #mapping_example()
    articulated_slam()
    '''
    robot_state = np.array([8.5,91.5,-np.pi/4])
    robot_cov = np.diag(np.array([100,100,np.pi]))
    for i in range(10):
        print robot_state
        print robot_cov
        robot_state,robot_cov=robot_motion_prop(robot_state,robot_cov,robot_input)
    '''

