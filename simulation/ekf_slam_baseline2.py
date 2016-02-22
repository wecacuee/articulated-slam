'''
Perform the actual SLAM on the landmarks and robot positions
'''
import landmarkmap3d_sim as landmarkmap
import cv2
import numpy as np
import scipy.stats as sp
import pdb
import utils_plot as up
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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

SIMULATEDDATA = True 
PLOTSIM = True

'''
Observation model of the robot - range bearing sensor
Page 207 - Eq: 7.12 - Probabilistic Robotics by Sebastian Thrun, Burgrad and Fox
'''
# Defining the non-linear observation model by first order approximation
# Needs two parameters, robot_state - The current robot state
# landmark_obs - Position of the landmark
def observation_model(robot_state,landmarks_pos):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Landmark position consists of (x,y)
    m_x = landmarks_pos[0];m_y = landmarks_pos[1]
    # Observation model for range
    r = np.sqrt((m_x-x)**2+(m_y-y)**2)
    # Observation model for bearing
    theta = np.arctan2(m_y-y,m_x-x)
    # Returning the mean prediction for now
    return np.array([r,theta])

# Defining the Jacobian of the observation model for EKF filtering for landmark position
def observation_jac(robot_state,landmarks_pos):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Landmark position consists of (x,y)
    m_x = landmarks_pos[0];m_y = landmarks_pos[1]
    # Equatiom 7.14 on Page 207 of Probabilistic Robotics
    q = (m_x-x)**2+(m_y-y)**2
    # Returning the H matrix for observation error propogation
    return np.array([[(m_x-x)/np.sqrt(q), (m_y-y)/np.sqrt(q)],
        [-(m_y-y)/q ,(m_x-x)/q ]])

# Defining the Jacobian of the observation model for EKF filtering for robot states
def observation_jac_robot(robot_state,landmarks_pos):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Landmark position consists of (x,y)
    m_x = landmarks_pos[0];m_y = landmarks_pos[1]
    # Equatiom 7.14 on Page 207 of Probabilistic Robotics
    q = (m_x-x)**2+(m_y-y)**2
    # Returning the H matrix for observation error propogation
    return np.array([[-(m_x-x)/np.sqrt(q), -(m_y-y)/np.sqrt(q) ,0 ],
        [(m_y-y)/q ,-(m_x-x)/q ,-1 ]])

# Robot bearing and range to x,y in cartesian frame given the robot state
def bearing_to_cartesian(obs,robot_state):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Observation is of range and bearing angle
    r = obs[0]; phi = obs[1]
     
    return np.array([x+r*np.cos(theta+phi),y+r*np.sin(theta+phi)])

def robot_to_world(robot_state,gen_obv):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]

    # v2.0 Using inverse rotation matrix to retrieve landmark world frame co    ordinates
    #R = np.array([[np.cos(theta), -np.sin(theta),0],
    #            [np.sin(theta), np.cos(theta),0],
    #            [0,0,1]])
    R = rodrigues([0,0,1],theta)
    # v2.0 
    # State is (x,y,0) since we assume robot is on the ground
    return R.T.dot(gen_obv)+ np.array([x,y,0])


# x,y in cartesian to robot bearing and range
def cartesian_to_bearing(obs,robot_state):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Observation is of x and y
    m_x = obs[0]; m_y = obs[1]
    return np.array([np.sqrt((m_x-x)**2+(m_y-y)**2),np.arctan2(m_y-y,m_x-x)])

#def threeptmap():
#    nframes = 100
#    map_conf = [# static
#               # dict(ldmks=np.array([[0, 0]]).T,
#               #      inittheta=0,
#               # # Where do we obtain the x and y from?   
#               #      initpos=[x, y],
#               #      deltheta=0,
#               #      delpos=[0,0]) 
#               # for x,y in zip([10] * 10 + range(10, 191, 20) + [190]*10 +
#               #                range(10, 191, 20),
#               #                range(10, 191, 20) + [190]*10 + 
#               #                range(10, 191, 20) + [10] * 10
#               #               )
#               #]+ [# prismatic
#               # dict(ldmks=np.array([[10,10]]).T,
#               #      inittheta=0,
#               #      initpos=[120,10],
#               #      deltheta=0,
#               #      delpos=[5,0])]
#               # # revolute
#               # dict(ldmks=np.array([[0,20]]).T, # point wrt rigid body frame
#               #      inittheta=np.pi,            # initial rotation
#               #      initpos=[160,40],            # origin of rigid body
#               #      deltheta=-10*np.pi/180,     # rotation per frame
#               #      delpos=[0,0])               # translation per frame
#               #]
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


def threeptmap3d():
    nframes = 110
    scale = 30. 
    #map_conf=   [#static
    #            dict(ldmks=np.array([[0,10,0,]]).T,
    #            initthetas=[0,0,0],
    #            initpos=[0,0,0],
    #            delthetas=[0,0,np.pi/10],
    #            delpos=[0,0,0])]    
    map_conf=   [#static
                dict(ldmks=np.array([[0,0,0,]]).T/scale,
                initthetas=[0,0,0],
                initpos=np.array([x,y,z])/scale,
                delthetas=[0,0,0],
                delpos=np.array([0,0,0])/scale)
                for x,y,z in zip([10]*10 + range(10,191,20)+[190]*10+range(10,191,20),
                                 range(10,191,20)+[190]*10+range(10,191,20)+[10]*10,
                                 [5]*10 + range(1,11,1)+[1]*10+range(1,11,1))]
                #]+[#Prismatic
                #dict(ldmks=np.array([[40,160,0]]).T/scale,
                #initthetas=[0,0,0],
                #initpos=np.array([0,0,0])/scale,
                #delthetas=[0,0,0],
                #delpos=np.array([1,0,0])/scale)
                #]+[#Revolute
                #dict(ldmks=np.array([[10,10,0]]).T/scale,
                #initthetas=[0,0,0],
                #initpos=np.array([130,130,0])/scale,
                #delthetas=[0,0,np.pi/5],
                #delpos=np.array([0,0,0])/scale)]

    lmmap = landmarkmap.map_from_conf(map_conf,nframes)
    robtraj = landmarkmap.robot_trajectory(
            np.array([[110,90,0],[140,60,0],[120,50,0],[110,90,0],[140,60,0]]) / scale,
            0.2, np.pi/50, True, 100/scale, np.array([40, 40])/scale, nframes)

    # For now static robot 
    #robtraj = landmarkmap.robot_trajectory(np.array([[0,0,0],[20,20,0]]),0.01,np.pi/10)
    #robtraj = landmarkmap.robot_trajectory(np.array([[60,140,0],[0,175,0],[-60,140,0],[-60,-140,0],[60,-140,0]]),0.2,np.pi/10)
    #robtraj = landmarkmap.robot_trajectory(np.array([[110,90,0],[40,175,0]]) / scale,0.1,np.pi/10)
    maxangle = 45*np.pi/180
    maxdist = 120/scale 
    return nframes,lmmap,robtraj,maxangle,maxdist


def visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state_2D,
                              slam_cov_2D, colors):
    thisframe = lmvis.genframe(ldmks, robview, colors)
    thisframe = lmvis.drawrobot(robview, thisframe)
    theta, width, height = up.ellipse_parameters_from_cov(slam_cov_2D,
                                                          volume=0.50)
    cv2.ellipse(thisframe, 
                tuple(map(np.int32, slam_state_2D * lmvis._scale)),
                tuple(np.int32(x * lmvis._scale) for x in (width/2, height/2)),
                theta, 0, 360,
                (0,0,255))
    cv2.imshow(lmvis._name, thisframe)
    cv2.waitKey(lmvis.frame_period)


'''
Propagates robot motion with two different models, one for linear velocity
and other one for a combination of linear.dot(gen_obv - np.array([x,y,0]))

Inputs are: 
Previous robot state,
covarinace in previous state,
actual robot input (translational and rotational component),
and time interval
'''
#def robot_motion_prop(prev_state,prev_state_cov,robot_input,delta_t=1):
#    # Robot input is [v,w]^T where v is the linear velocity and w is the rotational component
#    v = robot_input[0];w=robot_input[1];
#    # Robot state is [x,y,\theta]^T, where x,y is the position and \theta is the orientation
#    x = prev_state[0]; y=prev_state[1];theta = prev_state[2]
#    robot_state = np.zeros(3)
#    # Setting noise parameters, following Page 210 Chapter 7, Mobile Robot Localization of 
#    # Probabilistic Robotics book
#    alpha_1 = 0.1; alpha_2=0.05; alpha_3 = 0.05; alpha_4 = 0.1
#    # M for transferring noise from input space to state-space
#    M = np.array([[(alpha_1*(v**2))+(alpha_2*(w**2)),0],[0,(alpha_3*(v**2))+(alpha_4*(w**2))]])
#    
#    # Check if rotational velocity is close to 0
#    if (abs(w)<1e-4):
#        robot_state[0] = x+(v*delta_t*np.cos(theta))        
#        robot_state[1] = y+(v*delta_t*np.sin(theta))        
#        robot_state[2] = theta
#        # Derivative of forward dynamics model w.r.t previous robot state
#        G = np.array([[1,0,-v*delta_t*np.sin(theta)],\
#                [0,1,v*delta_t*np.cos(theta)],\
#                [0,0,1]])
#        # Derivative of forward dynamics model w.r.t robot input
#        V = np.array([[delta_t*np.cos(theta),0],[delta_t*np.sin(theta),0],[0,0]])
#    else:
#        # We have a non-zero rotation component
#        robot_state[0] = x+(((-v/w)*np.sin(theta))+((v/w)*np.sin(theta+w*delta_t)))
#        robot_state[1] = y+(((v/w)*np.cos(theta))-((v/w)*np.cos(theta+w*delta_t)))
#        robot_state[2] = theta+(w*delta_t)
#        G = np.array([[1,0,(v/w)*(-np.cos(theta)+np.cos(theta+w*delta_t))],\
#                [0,1,(v/w)*(-np.sin(theta)+np.sin(theta+w*delta_t))],\
#                [0,0,1]])
#        # Derivative of forward dynamics model w.r.t robot input
#        # Page 206, Eq 7.11
#        V = np.array([[(-np.sin(theta)+np.sin(theta+w*delta_t))/w,\
#                (v*(np.sin(theta)-np.sin(theta+w*delta_t)))/(w**2)+((v*np.cos(theta+w*delta_t)*delta_t)/w)],\
#                [(np.cos(theta)-np.cos(theta+w*delta_t))/w,\
#                (-v*(np.cos(theta)-np.cos(theta+w*delta_t)))/(w**2)+((v*np.sin(theta+w*delta_t)*delta_t)/w)],\
#                [0,delta_t]])
#    # Covariance in propagated state
#    state_cov = np.dot(np.dot(G,prev_state_cov),np.transpose(G))+np.dot(np.dot(V,M),np.transpose(V))
#    return robot_state,state_cov

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
    plt.figure('True Trajectory')
    true_robot_states = np.dstack(true_robot_states)[0]
    slam_robot_states = np.dstack(slam_robot_states)[0]
    #traj_ldmk1 = np.dstack(traj_ldmk1)[0]
    #traj_ldmk2 = np.dstack(traj_ldmk2)[0]
    #traj_ldmk3 = np.dstack(traj_ldmk3)[0]
    pdb.set_trace()
    plt.plot(true_robot_states[0],true_robot_states[1],'+-k',linestyle='dashed',label='True Robot trajectory',markersize=15.0)
    plt.figure('Slam Traj')
    plt.plot(slam_robot_states[0],slam_robot_states[1],'^-g',label='A-SLAM trajectory',markersize=15.0)

    #plt.plot(traj_ldmk1[0],traj_ldmk1[1],'*-g',linestyle='dotted',label='Prismatic joint',markers    ize=15.0)
    #plt.plot(traj_ldmk2[0],traj_ldmk2[1],'o-b',linestyle='dotted',label='Revolute joint',markersi    ze=10.0)
    #plt.plot(traj_ldmk3[0],traj_ldmk3[1],'^-r',label='Static joint',markersize=15.0)
    #plt.xticks([-2,0,2,4,6],fontsize=24)
    #plt.yticks([-2,0,2,4,6],fontsize=24)
    #plt.legend(loc=4,fontsize=24)
    plt.show()




'''
Performing EKF SLAM
Pass in optional parameter for collecting debug output for all the landmarks
'''
def slam(debug_inp=True):
    # Writing to file
    f_gt = open('gt_orig.txt','w')
    f_slam = open('slam_orig.txt','w')
    f= 300
    img_shape = (240, 320)
    camera_K_z_view = np.array([[f, 0, img_shape[1]/2], [0, f, img_shape[0]/2], [0, 0, 1]])
    # Getting the map
    #nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap3d()
    nframes, lmmap, robtraj, maxangle, maxdist = threeptmap3d()

    ldmk_am = dict(); # To keep a tab on the landmarks that have been previously seen

    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    # to get the landmarks with ids that are being seen by robot
    if SIMULATEDDATA:
        rob_obs_iter = landmarkmap.get_robot_observations(lmmap, robtraj, maxangle, maxdist,img_shape,camera_K_z_view,lmvis=None)
        motion_model = 'nonholonomic'
        csv = np.genfromtxt('expt_noise.csv',delimiter=',')
        count = 0
        robview = landmarkmap.RobotView(img_shape,camera_K_z_view,maxdist)
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

    #rob_obs_iter = list(rob_obs_iter)
    first_traj_pt = dict()
    #frame_period = lmvis.frame_period
    # EKF parameters for filtering

    # Initially we only have the robot state
    ( init_timestamp,_,rob_state_and_input,LDD,_) = rob_obs_iter.next()
    model = dict()
    slam_state =  np.array(rob_state_and_input[:3]) # \mu_{t} state at current time step
    # Covariance following discussion on page 317
    # Assuming that position of the robot is exactly known
    slam_cov = np.diag(np.ones(slam_state.shape[0])) # covariance at current time step
    ld_ids = [] # Landmark ids which will be used for EKF motion propagation step
    index_set = [slam_state.shape[0]] # To update only the appropriate indices of state and covariance 
    # Observation noise
    Q_obs = np.array([[5.0,0,0],[0,5.0,0],[0,0,5.0]])
    # For plotting
    obs_num = 0
    # Initial covariance for landmarks (x,y) position
    initial_cov = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]])
    # For error estimation in robot localization
    true_robot_states = []
    slam_robot_states = []
    ldmktracks = dict()
    xt2 = dict()
    xt1 = dict()

    if PLOTSIM:
        prob_plot1 = []
        prob_plot2 = []
        prob_plot3 = []
        traj_ldmk1 = []
        traj_ldmk2 = []
        traj_ldmk3 = []
        # Processing all the observations
        # We need to skip the first observation because it was used to initialize SLAM State
    for fidx, (timestamp,ids, rob_state_and_input, ldmks,ldmk_robot_obs) in enumerate(rob_obs_iter): 
        if not SIMULATEDDATA:
            if fidx<200:
                continue
            dt = timestamp - init_timestamp
            init_timestamp = timestamp
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state:', rob_state
        print 'Observations:', ldmk_robot_obs.shape

        #posdir = map(np.array, ([rob_state[0], rob_state[1]],
        #                        [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        #robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        
        # Following EKF steps now

        # First step is propagate : both robot state and motion parameter of any active landmark
        if not SIMULATEDDATA:
            slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],slam_cov[0:3,0:3],robot_input,dt*1e-9,motion_model=motion_model)
        else:   
            slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],slam_cov[0:3,0:3],robot_input)
            

        colors = []
        mm_probs = []
        # Collecting all the predictions made by the landmark
        ids_list = []
        # Processing all the observations
        for id,ldmk_rob_obv in zip(ids,np.dstack(ldmk_robot_obs)[0]):
            # Observation corresponding to current landmark is
            #obs = np.array([r, theta])
            if SIMULATEDDATA:
                # Adding noise to observations
                ldmk_rob_obv = ldmk_rob_obv + csv[count,:]
                count = count + 1

            if id not in xt2:
                xt2[id] = robot_to_world(slam_state[0:3],ldmk_rob_obv)                 
            else:
                '''
                Step 1: Process Observations to first determine the motion model of each landmark
                During this step we will use robot's internal odometry to get the best position of 
                external landmark as we can and try to get enough data to estimate landmark's motion
                model. 
                '''
                # For each landmark id, we want to check if the landmark has been previously seen
                # Setting default none value for the current landmark observation
                ldmk_am.setdefault(id,None)
                if ldmk_am[id] is None:
                    # Assign a static landmark id
                    ldmk_am[id] = 1 
                    ld_ids.append(id)
                    # Getting the current state to be added to the SLAM state (x,y) position of landmark
                    curr_ld_state = robot_to_world(slam_state,ldmk_rob_obv)
                    curr_ld_cov = initial_cov 
                    index_set.append(index_set[-1]+curr_ld_state.shape[0])
                    # Extend Robot state by adding the motion parameter of the landmark
                    slam_state = np.append(slam_state,curr_ld_state)
                    # Extend Robot covariance by adding the uncertainity corresponding to the 
                    # robot state
                    slam_cov = scipy.linalg.block_diag(slam_cov,curr_ld_cov) 

                    # Copy of x_{t-1}
                    xt1[id] = robot_to_world(slam_state[0:3],ldmk_rob_obv)

                else:
                    # This means this landmark is an actual observation that must be used for filtering
                    # the robot state as well as the motion parameter associated with the observed landmark

                    # Getting the predicted observation from the landmark articulated motion
                    # Getting the motion parameters associated with this landmark
                    curr_ind = ld_ids.index(id)
                    # Following steps from Table 10.2 from book Probabilistic Robotics
                    lk_pred = xt1[id] + (xt1[id] - xt2[id])

                    #lk_pred = robot_to_world(slam_state[0:3],ldmk_rob_obv)
                
                    R_temp = np.array([[np.cos(slam_state[2]), -np.sin(slam_state[2]),0],
                             [np.sin(slam_state[2]), np.cos(slam_state[2]),0],
                             [0,0,1]])
                    z_pred = R_temp.dot(lk_pred- np.append(slam_state[0:2],[0]))

                    #z_pred = R_temp.dot(slam_state[index_set[curr_ind]:index_set[curr_ind+1]] - np.append(slam_state[0:2],[0]))

                    #diff_vec = lk_pred-slam_state[0:2]
                    #q_val = np.dot(diff_vec,diff_vec)
                    #z_pred = np.array([np.sqrt(q_val),np.arctan2(diff_vec[1],diff_vec[0])-slam_state[2]])
                    # Getting the jacobian matrix 
                    H_mat = np.zeros((3,index_set[-1]))
                    # w.r.t robot state
                    curr_obs = ldmk_rob_obv
                    theta = slam_state[2]
                    H_mat[0,0:3] = np.array([-np.cos(theta),-np.sin(theta),-np.sin(theta)*curr_obs[0] + np.cos(theta)*curr_obs[1]])
                    H_mat[1,0:3] = np.array([np.sin(theta),-np.cos(theta),(-np.cos(theta)*curr_obs[0]) - (np.sin(theta)*curr_obs[1])])
                    H_mat[2,0:3] = np.array([0,0,0])
                    H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = np.dot(R_temp,np.array([[1],[1],[1]]))


                    # w.r.t landmark associated states
                    # Differentiation w.r.t landmark x and y first
                    #H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = (1.0/q_val)*np.array(\
                    #        [[np.sqrt(q_val)*diff_vec[0],np.sqrt(q_val)*diff_vec[1]],\
                    #        [-diff_vec[1],diff_vec[0]]])

                    # Innovation covariance
                    inno_cov = np.dot(H_mat,np.dot(slam_cov,H_mat.T))+Q_obs
                    # Kalman Gain
                    K_mat = np.dot(np.dot(slam_cov,H_mat.T),np.linalg.inv(inno_cov))
                    # Updating SLAM state
                    slam_state = slam_state+np.hstack((np.dot(K_mat,np.vstack((ldmk_rob_obv-z_pred)))))
                    # Updating SLAM covariance
                    slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)

                    # Updated model
                    xt2[id] = xt1[id]
                    xt1[id] = slam_state[index_set[curr_ind]:index_set[curr_ind+1]]
            # end of if else ldmk_am[id]

            #p1, p2, p3 = (0,0,1) # We are assuming everything to be static
            #color = np.int64((p1*rev_color
            #         + p2*pris_color
            #         + p3*stat_color))
            #color = color - np.min(color)
            #colors.append(color)
            ids_list.append(id)

        # end of loop over observations in single frame
        # Follow all the steps on
        for i, id in enumerate(ids):
            ldmktracks.setdefault(id, []).append(TrackedLdmk(timestamp, ldmk_robot_obs[:, i:i+1]))

        robview.set_robot_pos_theta((rob_state[0], rob_state[1], 0),
                rob_state[2])

        #img = lmvis.genframe(ldmks, ldmk_robot_obs=ldmk_robot_obs, robview = robview,colors=colors,SIMULATEDDATA=SIMULATEDDATA)
        #imgr = lmvis.drawrobot(robview, img)
        #imgrv = robview.drawtracks([ldmktracks[id] for id in ids],
        #        imgidx=robview.imgidx_by_timestamp(timestamp),
        #        colors=colors)
        #robview.visualize(imgrv)
        #lmvis.imshow_and_wait(imgr)

        #print "SLAM State for robot and landmarks is",slam_state
        obs_num = obs_num+1
        #up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        #visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
        #                          slam_cov[:2, :2], colors)
        R_temp_true = np.array([[np.cos(rob_state[2]), -np.sin(rob_state[2]),0],
                      [np.sin(rob_state[2]), np.cos(rob_state[2]),0],
                      [0,0,1]])
        R_temp = np.array([[np.cos(slam_state[2]), -np.sin(slam_state[2]),0],
                      [np.sin(slam_state[2]), np.cos(slam_state[2]),0],
                      [0,0,1]])
 
        quat_true = Rtoquat(R_temp_true)
        quat_slam = Rtoquat(R_temp)
        if fidx<1000:
            f_gt.write(str(fidx+1)+" "+str(rob_state[0])+" "+str(rob_state[1])+" "+str(0)+" "+str(quat_true[0])+" "+str(quat_true[1])+" "+str(quat_true[2])+" "+str(quat_true[3])+" "+"\n")
            f_slam.write(str(fidx+1)+" "+str(slam_state[0])+" "+str(slam_state[1])+" "+str(0)+" "+str(quat_slam[0])+" "+str(quat_slam[1])+" "+str(quat_slam[2])+" "+str(quat_slam[3])+" "+"\n")
 
        true_robot_states.append(rob_state)
        slam_robot_states.append(slam_state[0:3].tolist())
    # end of loop over frames

    if PLOTSIM: 
        plot_sim_res(PLOTSIM,prob_plot1,prob_plot2,prob_plot3,traj_ldmk1,traj_ldmk2,traj_ldmk3,true_robot_states,slam_robot_states)

    # Generating plots for paper

    #plt.figure('Trajectories')
    #true_robot_states = np.dstack(true_robot_states)[0]
    #slam_robot_states = np.dstack(slam_robot_states)[0]
    #plt.plot(true_robot_states[0],true_robot_states[1],'-k',linestyle='dashed',label='True Robot trajectory',markersize=15.0)
    #plt.plot(slam_robot_states[0],slam_robot_states[1],'^g',label='EKF SLAM trajectory',markersize=15.0)
    #plt.plot(prob_plot1[0],prob_plot1[1],'*g',label='Prismatic',markersize=15.0)
    #plt.plot(prob_plot2[0],prob_plot2[1],'ob',label='Revolute',markersize=15.0)
    #plt.plot(prob_plot3[0],prob_plot3[1],'^r',label='Static',markersize=15.0)
    #plt.yticks([-2,0,2,4,6],fontsize = 24)
    #plt.xticks([-2,0,2,4,6],fontsize = 24)
    #plt.legend(loc=4,fontsize=24)
    #plt.show()



    f_gt.close()
    f_slam.close()
    return (true_robot_states,slam_robot_states)


if __name__ == '__main__':
    # For reproducing the similar results with newer version of code
    slam()
    '''
    robot_state = np.array([8.5,91.5,-np.pi/4])
    robot_cov = np.diag(np.array([100,100,np.pi]))
    for i in range(10):
        print robot_state
        print robot_cov
        robot_state,robot_cov=robot_motion_prop(robot_state,robot_cov,robot_input)
    '''

