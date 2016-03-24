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
#from extracttrajectories import feature_odom_gt_pose_iter_from_bag

from itertools import imap, izip, tee as itr_tee, islice
from collections import namedtuple, deque
import csv

TrackedLdmk = namedtuple('TrackedLdmk', ['ts', 'pt3D'])
models_names = ['Revolute','Prismatic','Static']

PLOTSIM = True 


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

    
""" Function to display prior probabilities, true robot trajectory and SLAM robot trajectory """
def plot_sim_res(PLOTSIM,prob_plot1,traj_ldmk1,obs_ldmk1,true_robot_states,slam_robot_states, ldmk_am, model):
    #prob_plot1_dup = prob_plot1
    #prob_plot1 = np.dstack(prob_plot1)[0]

    #plt.figure('Object')
    #h1 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[0],'o-b',label='Revolute',linewidth=3.0,markersize=15.0)
    #h2 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[1],'*-g',label='Prismatic',linewidth=3.0,markersize=15.0)
    #h3 = plt.plot(range(len(prob_plot1_dup)),prob_plot1[2],'^-r',label='Static',linewidth = 3.0,markersize=15.0)
    #plt.xlabel('Number of frames',fontsize=24)
    #plt.ylabel('Probability',fontsize=24)
    #plt.xticks([0,2,4,6,8,10,12,14,16,18],fontsize=24)
    #plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=24)
    #plt.legend(loc=3,fontsize=24)

    #plt.figure('Robot Trajectories')
    #true_robot_states = np.dstack(true_robot_states)[0]
    #slam_robot_states = np.dstack(slam_robot_states)[0]
    #plt.plot(true_robot_states[0],true_robot_states[1],'+-k',linestyle='dashed',label='True Robot trajectory',markersize=15.0)
    #plt.plot(slam_robot_states[0],slam_robot_states[1],'^-g',label='A-SLAM trajectory',markersize=15.0)
    #plt.xticks([-2,0,2,4,6],fontsize=24)
    #plt.yticks([-2,0,2,4,6],fontsize=24)
    #plt.legend(loc=4,fontsize=24)


    fig = plt.figure('Landmark Trajectory')
    traj_ldmk1 = np.dstack(traj_ldmk1)[0]
    obs_ldmk1 = np.dstack(obs_ldmk1)[0]
    ax = fig.add_subplot(111,projection='3d')
    # Strict case of single landmark and revolute model
    if model[1] == 0:
        config_pars = ldmk_am.config_pars
        vec1 = config_pars['vec1']
        vec2 = config_pars['vec2']
        center = config_pars['center']
        center3D = center[0]*vec1 + center[1]*vec2 + ldmk_am.plane_point
        axis_vec = np.cross(vec1, vec2)
        radius = config_pars['radius']
        
        # Drawing rev axis
        norm = np.linalg.norm
        size= 1
        pt13D = center3D + axis_vec * size/ norm(axis_vec)
        pt23D = center3D - axis_vec * size/ norm(axis_vec)
        #pt1 = self.projected_world(pt13D.reshape(-1, 1))
        #pt2 = self.projected_world(pt23D.reshape(-1, 1))
        ax.plot([pt13D[0],pt23D[0]],[pt13D[1],pt23D[1]],[pt13D[2],pt23D[2]],color='b')
            



    
    ## SLAM robot trajectory
    #temp = 'NULL'
    #for xs,ys,zs in zip(slam_robot_states[0],slam_robot_states[1],slam_robot_states[2]):
    #    ax.scatter(xs,ys,zs,c='g',marker='^')
    #    if temp == 'NULL':
    #        temp = [xs,ys,zs]
    #    else:
    #        ax.plot([temp[0],xs],[temp[1],ys],[temp[2],zs],color='k')
    #        temp = [xs,ys,zs]
    ## True robot trajectory
    #temp = 'NULL'
    #for xs,ys,zs in zip(true_robot_states[0],true_robot_states[1],true_robot_states[2]):
    #    ax.scatter(xs,ys,zs,c='k',marker='+')
    #    if temp == 'NULL':
    #        temp = [xs,ys,zs]
    #    else:
    #        ax.plot([temp[0],xs],[temp[1],ys],[temp[2],zs],color='k')
    #        temp = [xs,ys,zs]
   
    # Predicted landmark trajectory
    temp = 'NULL'
    for xs,ys,zs in zip(traj_ldmk1[0],traj_ldmk1[1],traj_ldmk1[2]):
        ax.scatter(xs,ys,zs,c='b',marker='o')
        if temp == 'NULL':
            temp = [xs,ys,zs]
        else:
            ax.plot([temp[0],xs],[temp[1],ys],[temp[2],zs],color='r')
            temp = [xs,ys,zs]
    # Observed landmark trajectory
    temp = 'NULL'
    for xs,ys,zs in zip(obs_ldmk1[0],obs_ldmk1[1],obs_ldmk1[2]):
        ax.scatter(xs,ys,zs,c='r',marker='^')
        if temp == 'NULL':
            temp = [xs,ys,zs]
        else:
            ax.plot([temp[0],xs],[temp[1],ys],[temp[2],zs],color='r')
            temp = [xs,ys,zs]

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend()
    
    plt.show()



""" April Tags to simulated data format """
class ARtag(object):
    def __init__(self):
        self.filename = "../CamDataRevDoorCorrect.txt"
        self.Hom = []
        for item in range(36):
            self.Hom.append(np.identity(4))
    
    def formatter(self,sp,empty_F=False):
        if not empty_F:
            return np.asarray([np.asarray([sp[0]],dtype='float64'),np.asarray([int(sp[1])]),
                    np.asarray([float(0.0),float(0.0),float(0.0),float(0.0),float(0.0)]),
                    np.asarray([sp[4],sp[5],sp[6]],dtype='float64'),np.asarray([sp[4],sp[5],sp[6]],dtype='float64')])
        else:
            return np.asarray([np.asarray([sp[0]],dtype='float64'),[],
                    np.asarray([float(0.0),float(0.0),float(0.0),float(0.0),float(0.0)]),
                    np.asarray([]),np.asarray([])])

    def homographyGen(self,sp):
        Rot = np.identity(4)
        gamma,pitch,roll = [float(sp[6]),float(sp[7]),float(sp[8])]
        Rx = np.asarray([[1.0,0.0,0.0],
                        [0.0, np.cos(roll),-np.sin(roll)],
                        [0.0,np.sin(roll),np.cos(roll)]])
        Ry = np.asarray([[np.cos(pitch),0.0,np.sin(pitch)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(pitch),0.0,np.cos(pitch)]])
        Rz = np.asarray([[np.cos(gamma),0.0,np.sin(gamma)],
                        [np.sin(gamma), np.cos(gamma), 0.0],
                        [0.0,0.0,1.0]])
        
        Rot[0:3,0:3] = Rz.dot(Ry.dot(Rx)) 
        Rot[0:3,-1]  = np.asarray([sp[3],sp[4],sp[5]],dtype='float')  
        
        self.Hom[int(sp[1])] = self.Hom[int(sp[1])].dot(Rot)
        

    def processData(self):
        fp = open(self.filename,'r')
        prev_ts = 'NULL'
        op = []
        prev_ts = 'NULL'

        for line in fp:
            sp = line.split(',')
            timestamp = sp[0]
            if prev_ts == 'NULL' and sp[1]!='INF':
                prev_ts = timestamp
                self.homographyGen(sp)
                op.append(self.formatter(sp))
        
            elif prev_ts == timestamp:
                self.homographyGen(sp)
                op.append(self.formatter(sp))
                prev_ts = timestamp
        
            elif prev_ts != timestamp:
                if len(op)!=0:
                    X = np.dstack(op)
                    yield X[0].tolist()
                op = []
                if sp[1]=='INF':
                    op.append(self.formatter(sp,True))
                else:
                    self.homographyGen(sp)
                    op.append(self.formatter(sp))
                prev_ts = timestamp

        X = np.dstack(op)
        yield X[0].tolist()

                

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
    m_thresh = 0.40 # Choose the articulation model with greater than this threhsold's probability
    
    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ekf_map_id = dict(); # Formulating consistent EKF mean and covariance vectors
    
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    
    # Handle simulated and real data , setup required variables
    motion_model = 'nonholonomic'
    art = ARtag()
    rob_obs_iter = art.processData()
    chk_F = False

    lmvis = landmarkmap.LandmarksVisualizer([0,0], [7, 7], frame_period=10,
                                            imgshape=(700, 700))
    
    # EKF parameters for filtering
    first_traj_pt = dict()
    # Initially we only have the robot state
    (init_timestamp, _, rob_state_and_input, _, _) = rob_obs_iter.next() 
    rob_state_and_input = rob_state_and_input[0]
    
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
        obs_ldmk1 = []

    # Processing all the observations
    for fidx,(timestamp, ids,rob_state_and_input, ldmks, ldmk_robot_obs) in enumerate(rob_obs_iter):
        
        # Need more data to skip and find model
        if fidx%1!=0:
            continue
        if len(ids)!=0:
            rob_state_and_input = rob_state_and_input[0]
        if len(ldmk_robot_obs) != 0:
            ldmk_robot_obs = ldmk_robot_obs[0]
        #if fidx < 200:
        #    continue
        dt = timestamp[0] - init_timestamp[0]
        init_timestamp = timestamp
        
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state and inputs:', rob_state, robot_input
        print 'Observations size:',ldmk_robot_obs.shape
        
        # Following EKF steps now

        # First step is propagate : both robot state and motion parameter of any active landmark
        slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],
                                                            slam_cov[0:3,0:3],
                                                            robot_input,dt,
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
        if len(ids[0]) !=0:
            
            for id, ldmk_rob_obv in zip(ids,np.dstack(ldmk_robot_obs)[0]):
                id = id[0]
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
                if PLOTSIM == 1 and id ==1:
                    obs_ldmk1.append(ldmk_rob_obv.copy())

                # For each landmark id, we want to check if the motion model has been estimated
                if ldmk_am[id] is None:
                    
                    motion_class.process_inp_data([0,0], slam_state[:3],ldmk_rob_obv,first_traj_pt[id])
                    # Still need to estimate the motion class
                    # Check if the model is estimated
                    print motion_class.prior        
                    if sum(motion_class.prior>m_thresh)>0:
                        model[id] = np.where(motion_class.prior>m_thresh)[0]	
                        print("INFO: Model estimated %s " % models_names[int(model[id])])
                        print id
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
                    lk_pred = ldmk_am[id].predict_model(slam_state[index_set[curr_ind]:index_set[curr_ind+1]])
                    print "Obs: ",ldmk_rob_obv
                    print "Pred: ",lk_pred
                    if PLOTSIM and id == 1:
                        traj_ldmk1.append(lk_pred.copy())
                    

                    ld_preds.append(lk_pred)
                    ld_ids_preds.append(curr_ind)

		            # v2.0
                    R_temp = rodrigues([0,0,1],slam_state[2]).T
                    pos_list = np.ndarray.tolist(slam_state[0:2])
                    pos_list.append(0.0)
                    # To match R_w2c matrix
                    z_pred = R_temp.dot(lk_pred-np.array(pos_list)) 
                       
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
                    #print slam_state[0:3],id
                    # Updating SLAM covariance
                    slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)
                # end of if else ldmk_am[id]
                
                if PLOTSIM:
                    # Assuming single landmark exists
                    if id == 1:
                        prob_plot1.append(motion_class.prior.copy())

                mm_probs.append(motion_class.prior.copy())

                motion_class = ldmk_estimater[id]
                p1, p2, p3 = motion_class.prior[:3]
                color = np.int64((p1*rev_color
                         + p2*pris_color
                         + p3*stat_color))
                color = color - np.min(color)
                colors.append(color)
                ids_list.append(id)

        # end of loop over observations in single frame

        #if PLOTSIM:
        #    if 1 in ldmk_am.keys() and ldmk_am[1] is not None and not chk_F:
        #        import pdb; pdb.set_trace()
        #        motion_class1 = ldmk_estimater.setdefault(1, mm.Estimate_Mm())
        #        #lk_pred = ldmk_am[1].predict_model(slam_state[index_set[ld_ids.index(1)]:index_set[ld_ids.index(1)]+1])
        #        lk_pred = ldmk_am[1].predict_model(motion_class1.means[int(model[1])])
        #        traj_ldmk1.append(lk_pred.copy())
        #chk_F = False

        
        # Follow all the steps on
        #print "SLAM State for robot and landmarks is",slam_state
        #obs_num = obs_num+1
        #print 'motion_class.priors', mm_probs
        #print 'ids:', ids_list
        #print 'colors:', colors
        ##up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        #visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
        #                          slam_cov[:2, :2], colors,obs_num)

        #for i, id in enumerate(ids):
        #    ldmktracks.setdefault(id, []).append(
        #        TrackedLdmk(timestamp, ldmk_robot_obs[:, i:i+1]))

        ##Handle new robot view code appropriately
        #robview.set_robot_pos_theta((rob_state[0], rob_state[1], 0),
        #                            rob_state[2]) 
        #assert ldmk_robot_obs.shape[1] == len(colors), '%d <=> %d' % (
        #    ldmk_robot_obs.shape[1], len(colors))
        
        ##Img = lmvis.genframe(ldmks, ldmk_robot_obs=ldmk_robot_obs, robview = robview,colors=colors,SIMULATEDDATA=SIMULATEDDATA)
        ##Imgr = lmvis.drawrobot(robview, img)
        ##Imgrv = robview.drawtracks([ldmktracks[id] for id in ids],
        ##                           imgidx=robview.imgidx_by_timestamp(timestamp),
        ##                           colors=colors)


        ##if not SIMULATEDDATA:

        ##    for id in ld_ids:
        ##        if model[id] == 0:
        ##            config_pars = ldmk_am[id].config_pars
        ##            vec1 = config_pars['vec1']
        ##            vec2 = config_pars['vec2']
        ##            center = config_pars['center']
        ##            center3D = center[0]*vec1 + center[1]*vec2 + ldmk_am[id].plane_point
        ##            axis_vec = np.cross(vec1, vec2)
        ##            radius = config_pars['radius']
        ##            imgrv = robview.drawrevaxis(imgrv, center3D, axis_vec, radius, rev_color)
        ##            imgr = lmvis.drawrevaxis(imgr, center3D, axis_vec, radius,
        ##                                     rev_color)

        ##robview.visualize(imgrv)
        ##lmvis.imshow_and_wait(imgr)
        #    

        quat_true = Rtoquat(rodrigues([0,0,1],rob_state[2]))
        quat_slam = Rtoquat(rodrigues([0,0,1],slam_state[2]))    
        if fidx < 1000:
            f_gt.write(str(fidx+1)+" "+str(rob_state[0])+" "+str(rob_state[1])+" "+str(0)+" "+str(quat_true[0])+" "+str(quat_true[1])+" "+str(quat_true[2])+" "+str(quat_true[3])+" "+"\n")
            f_slam.write(str(fidx+1)+" "+str(slam_state[0])+" "+str(slam_state[1])+" "+str(0)+" "+str(quat_slam[0])+" "+str(quat_slam[1])+" "+str(quat_slam[2])+" "+str(quat_slam[3])+" "+"\n")
        true_robot_states.append(rob_state)
        slam_robot_states.append(slam_state[0:3].tolist())

        #obs_num = obs_num + 1
        print 'SLAM state:',slam_state[0:3]
    # end of loop over frames
    
    if PLOTSIM:
        # Plot experimental results
        plot_sim_res(PLOTSIM,prob_plot1,traj_ldmk1,obs_ldmk1,true_robot_states,slam_robot_states,ldmk_am[1],model)

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
    #temp()
    articulated_slam()
    '''
    robot_state = np.array([8.5,91.5,-np.pi/4])
    robot_cov = np.diag(np.array([100,100,np.pi]))
    for i in range(10):
        print robot_state
        print robot_cov
        robot_state,robot_cov=robot_motion_prop(robot_state,robot_cov,robot_input)
    '''

