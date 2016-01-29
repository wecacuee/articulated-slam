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

from itertools import imap, izip

SIMULATEDDATA = False
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


def threeptmap3d():
    nframes = 100
    scale = 30.
    #map_conf=   [#static
    #            dict(ldmks=np.array([[0,0,0,]]).T / scale,
    #            initthetas=[0,0,0],
    #            initpos=[0,10,0],
    #            delthetas=[0,0,0*np.pi/10],
    #            delpos=[0.5,0,0])]
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
                dict(ldmks=np.array([[0,160,0]]).T / scale,
                initthetas=[0,0,0],
                initpos=np.array([0,0,0]) / scale,
                delthetas=[0,0,0],
                delpos=np.array([2,0,0]) / scale)
                ]+[#Revolute
                dict(ldmks=np.array([[10,10,0]]).T / scale,
                initthetas=[0,0,0],
                initpos=np.array([110,90,0]) / scale,
                delthetas=[0,0,np.pi/50],
                delpos=np.array([0,0,0]) / scale)]
    
    lmmap = landmarkmap.map_from_conf(map_conf,nframes)
    # For now static robot 
    #robtraj = landmarkmap.robot_trajectory(np.array([[0,0,0],[20,20,0]]),0.01,np.pi/10)
    robtraj = landmarkmap.robot_trajectory(
        np.array([[110,90,0],[140,60,0],[120,50,0],[110,90,0],[140,60,0]]) / scale,
        0.2, np.pi/10, True, 20/scale, np.array([20, 20])/scale, nframes)
    #robtraj = landmarkmap.robot_trajectory(np.array([[110,90,0],[40,175,0]]) / scale,0.1,np.pi/10)
    maxangle = 45*np.pi/180
    maxdist = 120 / scale
    return nframes,lmmap,robtraj,maxangle,maxdist

def Rtoquat(R):
	qw = np.sqrt(1+R[0,0,]+R[1,1]+R[2,2])/2.0
	qx = (R[2,1] - R[1,2])/4/qw
	qy = (R[0,2] - R[2,0])/4/qw
	qz = (R[1,0] - R[0,1])/4/qw
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

class RealDataToSimulatedFormat(object):
    def __init__(self, K):
        # (z_view) real_k assumes Z in viewing direction, Y-axis downwards
        # (x_viwe) Simulated data format assumes X in viewing direction, Z -axis upwards
        R_x_view_2_z_view = rodrigues([0, 1, 0], -np.pi/2).dot(
            rodrigues([1, 0, 0], np.pi/2))
        K_x_view = K.dot(R_x_view_2_z_view)
        self.camera_K_x_view = K_x_view
        self._first_gt_pose = None
        self._all_ldmks = np.array([])

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
        max_track_id = np.maximum(np.max(track_ids), len(self._all_ldmks))
        all_ldmks = np.zeros((ldmks_world.shape[0], max_track_id + 1))
        if (len(self._all_ldmks)):
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
        assert abs(xyz[2]) < 0.05, 'xyz[2] - z0 = %f; z0 = %f' % (xyz[2], z0)
        assert abs(abg[0]) < 0.05, 'abg[0] = %f' % abg[0]
        assert abs(abg[1]) < 0.05, 'abg[1] = %f' % abg[1]
        return (xyz[:2], abg[2])

    def real_data_to_required_format(self, ts_features_odom_gt):
        ts, features_odom_gt = ts_features_odom_gt
        tracked_pts, odom, gt_pose = features_odom_gt
        track_ids, points, depths  = zip(*tracked_pts)
        (linvel2D, angvel2D) = self.unpack_odom(odom)
        (xy, theta) = self.unpack_gt_pose(gt_pose)
        ldmk_robot_obs = self.compute_ldmks_robot_pose(points, depths)
        all_ldmks = self.prepare_ldmks_all_world(theta, ldmk_robot_obs,
                                     xy, track_ids)

        return (ts, track_ids, [xy[0], xy[1], theta,
                            np.asarray(linvel2D), angvel2D],
                all_ldmks, ldmk_robot_obs)

'''
Performing Articulated SLAM
Pass in optional parameter for collecting debug output for all the landmarks
'''
def articulated_slam(debug_inp=True):
    # Writing to file variables
    f_gt = open('gt.txt','w')
    f_slam = open('slam.txt','w')    
    img_shape = (240, 320)
    f = 275
    camera_K_z_view = np.array([[f, 0, img_shape[1]/2.], 
                       [0, f, img_shape[0]/2.],
                       [0,   0,   1]])
    timeseries_data_file = "/home/vikasdhi/mid/articulatedslam/2016-01-22/rev_2016-01-22-13-56-28/extracttrajectories_GFTT_SIFT_odom_gt_timeseries.txt"
    timeseries_dir = os.path.dirname(timeseries_data_file)
    image_file_fmt = os.path.join(timeseries_dir, "img/frame%04d.png")
    timestamps_file = os.path.join(timeseries_dir, 'img/timestamps.txt')

    # Motion probability threshold
    m_thresh = 0.60 # Choose the articulation model with greater than this threhsold's probability
    
    # Getting the map
    nframes,lmmap,robtraj,maxangle,maxdist = threeptmap3d()
    robview = landmarkmap.RobotView(img_shape, camera_K_z_view, maxdist,
                                   image_file_format=image_file_fmt,
                                   timestamps_file=timestamps_file)
    #nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ekf_map_id = dict(); # Formulating consistent EKF mean and covariance vectors
    
    # Commenting old visualization code component	
    #rev_color, pris_color, stat_color = [np.array(l) for l in (
    #    [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    
    # to get the landmarks with ids that are being seen by robot (Need to modify to match 3D viewing cone)
    # Handle viewable observations based on new code
    if SIMULATEDDATA:
        rob_obs_iter = landmarkmap.get_robot_observations(lmmap, robtraj,
                                                          maxangle, maxdist,
                                                          img_shape,
                                                          camera_K_z_view,
                                                  # Do not pass visualizer to
                                                  # disable visualization
                                                  lmvis=None)
        motion_model = 'nonholonomic'
    else:
        serializer = dataio.FeaturesOdomGTSerializer()
        datafile = open(timeseries_data_file)
        timestamps = serializer.init_load(datafile)
        data_iter = serializer.load_iter(datafile)
        VISDEBUG = 0
        if VISDEBUG:
            plotables = dict()
            for i, (features, odom, gt_pose) in enumerate(data_iter):
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
        rob_obs_iter = imap(
            formatter.real_data_to_required_format,
            izip(timestamps, data_iter))

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
    # Commenting old visualization code compoenents 
    #For plotting
    obs_num = 0
    # For error estimation in robot localization
    true_robot_states = []
    slam_robot_states = []
    # Processing all the observations
    # v1.0 Need to update to v2.0 with no rs and thetas
    # v2.0 Expected format
    for fidx,(timestamp, ids,rob_state_and_input, ldmks, ldmk_robot_obs) in enumerate(rob_obs_iter):    
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state and inputs:', rob_state, robot_input 
        # v1.0
        #print 'Observations:', zip(rs, thetas, ids)
        # v2.0 
        print 'Observations:',ldmk_robot_obs
        
        # Handle new robot view code appropriately
        robview.set_robot_pos_theta((rob_state[0], rob_state[1], 0),
                                    rob_state[2]) 
        img = lmvis.genframe(ldmks, robview)
        imgr = lmvis.drawrobot(robview, img)
        imgrv = robview.drawlandmarks(ldmk_robot_obs,
                                      imgidx=robview.imgidx_by_timestamp(timestamp))
        robview.visualize(imgrv)
        lmvis.imshow_and_wait(imgr)
        
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
    
        # Processing all the observations
        # v2.0
        for id, ldmk_rob_obv in zip(ids,np.dstack(ldmk_robot_obs)[0]):
            if id not in first_traj_pt:
                first_traj_pt[id] = ldmk_rob_obv
           
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
                motion_class.process_inp_data([0,0], slam_state,ldmk_rob_obv,first_traj_pt[id])
                # Still need to estimate the motion class
                # Check if the model is estimated
                if sum(motion_class.prior>m_thresh)>0:
                    print("INFO: Model estimated\n")
                    model[id] = np.where(motion_class.prior>m_thresh)[0]	
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
                R_temp = np.array([[np.cos(slam_state[2]), -np.sin(slam_state[2]),0],
                        [np.sin(slam_state[2]), np.cos(slam_state[2]),0],
                        [0,0,1]])


                pos_list = np.ndarray.tolist(slam_state[0:2])
                pos_list.append(0.0)
                z_pred = R_temp.dot(lk_pred-np.array(pos_list))
                
                # v2.0 : New definition
                H_mat = np.zeros((3,index_set[-1]))
                # v2.0 Need to modify based on 3D rotation matrix jacobian: Need to find the updated theta value
                curr_obs = ldmk_rob_obv
                theta = slam_state[2]
                H_mat[0,0:3] = np.array([-np.cos(theta),-np.sin(theta),-np.sin(theta)*curr_obs[0]+ np.cos(theta)*curr_obs[1]])
                H_mat[1,0:3] = np.array([np.sin(theta),-np.cos(theta),(-np.cos(theta)*curr_obs[0])-(np.sin(theta)*curr_obs[1])])
                H_mat[2,0:3] = np.array([0,0,0])

                #H_mat[0,0:3] = np.array([1,0,-np.sin(theta)*curr_obs[0] - np.cos(theta)*curr_obs[1]])
                #H_mat[1,0:3] = np.array([0,1,(np.cos(theta)-np.sin(theta))*curr_obs[0] - (np.cos(theta)+np.sin(theta))*curr_obs[1]])
                #H_mat[2,0:3] = np.array([0,0,(np.sin(theta)+np.cos(theta))*curr_obs[0] + (np.cos(theta)+np.sin(theta))*curr_obs[1]])
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
            mm_probs.append(motion_class.prior)

            # commenting out old visualization code
            #motion_class = ldmk_estimater[id]
            #p1, p2, p3 = motion_class.prior[:3]
            #color = np.int64((p1*rev_color
            #         + p2*pris_color
            #         + p3*stat_color))
            #color = color - np.min(color)
            #colors.append(color)
            ids_list.append(id)

        # end of loop over observations in single frame
                
        # Follow all the steps on
        #print "SLAM State for robot and landmarks is",slam_state
        #obs_num = obs_num+1
        #print 'motion_class.priors', mm_probs
        #print 'ids:', ids_list
        #print 'colors:', colors
        ##up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        #visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
        #                          slam_cov[:2, :2], colors,obs_num)
        R_temp_true = np.array([[np.cos(-rob_state[2]), -np.sin(-rob_state[2]),0],
                      [np.sin(-rob_state[2]), np.cos(-rob_state[2]),0],
                      [0,0,1]]) 
        R_temp = np.array([[np.cos(-slam_state[2]), -np.sin(-slam_state[2]),0],
                      [np.sin(-slam_state[2]), np.cos(-slam_state[2]),0],
                      [0,0,1]])

        quat_true = Rtoquat(R_temp_true)
        quat_slam = Rtoquat(R_temp)    
        f_gt.write(str(fidx+1)+" "+str(rob_state[0])+" "+str(rob_state[1])+" "+str(0)+" "+str(quat_true[0])+" "+str(quat_true[1])+" "+str(quat_true[2])+" "+str(quat_true[3])+" "+"\n")
        f_slam.write(str(fidx+1)+" "+str(slam_state[0])+" "+str(slam_state[1])+" "+str(0)+" "+str(quat_slam[0])+" "+str(quat_slam[1])+" "+str(quat_slam[2])+" "+str(quat_slam[3])+" "+"\n")
        true_robot_states.append(rob_state)
        slam_robot_states.append(slam_state[0:3].tolist())
        obs_num = obs_num + 1
        print 'SLAM state:',slam_state
    # end of loop over frames

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

