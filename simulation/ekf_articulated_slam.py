'''
Performs the main goal of this project, being able to use articulation
inside SLAM, by first
- Getting few observations and estimating the articulated models
- Using the motion parameters of those models to propagate the landmarks and the robot
- Using observations to update the EKF state (consisting of robot pose and motion parameters)

- Code is similar to threeptmap.py
'''

import numpy as np
import cv2
import landmarkmap
import estimate_mm as mm # To figure out the right motion model
import pdb
import utils_plot as up
import scipy.linalg
import matplotlib.pyplot as plt

def threeptmap():
    nframes = 100
    map_conf = [# static
                dict(ldmks=np.array([[0, 0]]).T,
                     inittheta=0,
                     initpos=[x, y],
                     deltheta=0,
                     delpos=[0,0]) 
                for x,y in zip([10] * 10 + range(10, 191, 20) + [190]*10 +
                               range(10, 191, 20),
                               range(10, 191, 20) + [190]*10 + 
                               range(10, 191, 20) + [10] * 10
                              )
               ] + [# prismatic
                dict(ldmks=np.array([[10,10]]).T,
                     inittheta=0,
                     initpos=[120,10],
                     deltheta=0,
                     delpos=[5,0]),
                # revolute
                dict(ldmks=np.array([[0,20]]).T, # point wrt rigid body frame
                     inittheta=np.pi,            # initial rotation
                     initpos=[160,40],            # origin of rigid body
                     deltheta=-10*np.pi/180,     # rotation per frame
                     delpos=[0,0])               # translation per frame
               ]
    # Possible issue: Revolute joint appears as 160,20 in the first frame. Is that right???

    lmmap = landmarkmap.map_from_conf(map_conf, nframes)
    lmvis = landmarkmap.LandmarksVisualizer([0,0], [200, 200], frame_period=-1,
                                         scale=3)
    robtraj = landmarkmap.robot_trajectory(np.array([[110, 90], [140,60],
                                                     [120,50], [110, 90], 
                                                     [140, 60]]),
                                           5, np.pi/10)
    # angle on both sides of robot dir
    maxangle = 45*np.pi/180
    # max distance in pixels
    maxdist = 120

    return nframes, lmmap, lmvis, robtraj, maxangle, maxdist

def visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state_2D,
                              slam_cov_2D, colors,obs_num):
    thisframe = lmvis.genframe(ldmks, robview, colors)
    thisframe = lmvis.drawrobot(robview, thisframe)
    theta, width, height = up.ellipse_parameters_from_cov(slam_cov_2D,
                                                          volume=0.50)
    cv2.ellipse(thisframe, 
                tuple(map(np.int32, slam_state_2D * lmvis._scale)),
                tuple(np.int32(x * lmvis._scale) for x in (width/2, height/2)),
                theta, 0, 360,
                (0,0,255),2)
    cv2.imshow(lmvis._name, thisframe)
    filename = '../media/ekf_frame%04d.png' % obs_num 
    print 'Writing frame to %s' % filename
    cv2.imwrite(filename,thisframe)
    cv2.waitKey(lmvis.frame_period)

def mapping_example():
    nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    # to get the landmarks with ids that are being seen by robot
    rob_obs_iter = landmarkmap.get_robot_observations(
        lmmap, robtraj, maxangle, maxdist, 
                                              # Do not pass visualizer to
                                              # disable visualization
                                              lmvis=None)
    frame_period = lmvis.frame_period
    for fidx, (rs, thetas, ids, rob_state_and_input, ldmks) in enumerate(rob_obs_iter): 
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot state:', rob_state
        print 'Observations:', zip(rs, thetas)
        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        colors = []
        mm_probs = []
        for r, theta, id in zip(rs, thetas, ids):
            motion_class = ldmk_estimater.setdefault(id, mm.Estimate_Mm())
            obs = [r, theta]
            motion_class.process_inp_data(obs, rob_state)
            color = np.int64((motion_class.prior[0]*rev_color 
                     + motion_class.prior[1]*pris_color
                     + motion_class.prior[2]*stat_color))
            color = color - np.min(color)
            colors.append(color)
            mm_probs.append(motion_class.prior)

        img = lmvis.genframe(ldmks, robview, colors=colors)
        img = lmvis.drawrobot(robview, img)

        # Draw estimated trajectory
        for id in ids:
            motion_class = ldmk_estimater[id]
            # Plot trajectory
            if motion_class.prior[0] > 0.9:
                # revolute
                center, radius, theta_0, omega = motion_class.am[0].get_revolute_par()
                #cv2.ellipse(img, centerpt,
                #            np.int64(radius)*lmvis._scale, 0, theta_0, theta_0 + omega*10,
                #           color=rev_color, thickness=1*lmvis._scale)
                for i in range(10):
                    angle = theta_0 + i * omega
                    pt1 = center + radius * np.array([np.cos(angle),
                                               np.sin(angle)])
                    pt1 = tuple(np.int64(pt1)*lmvis._scale)
                    pt2 = center + radius * np.array([np.cos(angle+np.pi/180),
                                               np.sin(angle+np.pi/180)])
                    pt2 = tuple(np.int64(pt2)*lmvis._scale)
                    if np.all(pt1 <= img.shape) and np.all(pt2 <= img.shape):
                        cv2.line(img, pt1, pt2, color=rev_color, thickness=1*lmvis._scale)
            elif motion_class.prior[1] > 0.9:
                # prismatic
                x0, delx = motion_class.am[1].get_prismatic_par()
                pt1 = tuple(np.int64(x0)*lmvis._scale)
                pt2 = tuple(np.int64(x0+delx*10)*lmvis._scale)
                if np.all(pt1 <= img.shape) and np.all(pt2 <= img.shape):
                    cv2.line(img, pt1, pt2, color=pris_color, thickness=1*lmvis._scale)
        #colors
        print 'motion_class.priors', mm_probs
    
    
        if fidx in [0,2,4,6,8,10,12,14]:
            filename = '../media/frame%04d.png' % fidx
            print 'Writing to %s' % filename
            cv2.imwrite(filename, img)

        cv2.imshow(lmvis._name, img)
        keyCode = cv2.waitKey(frame_period)
        if keyCode in [1048608, 32]: # space
            frame_period = lmvis.frame_period if frame_period == -1 else -1
        elif keyCode != -1:
            print 'Keycode = %d' % keyCode

'''
Propagates robot motion with two different models, one for linear velocity
and other one for a combination of linear and rotational velocity
Inputs are: 
Previous robot state,
covarinace in previous state,
actual robot input (translational and rotational component),
and time interval
'''
def robot_motion_prop(prev_state,prev_state_cov,robot_input,delta_t=1):
    # Robot input is [v,w]^T where v is the linear velocity and w is the rotational component
    v = robot_input[0];w=robot_input[1];
    # Robot state is [x,y,\theta]^T, where x,y is the position and \theta is the orientation
    x = prev_state[0]; y=prev_state[1];theta = prev_state[2]
    robot_state = np.zeros(3)
    # Setting noise parameters, following Page 210 Chapter 7, Mobile Robot Localization of 
    alpha_1 = 0.1; alpha_2=0.05; alpha_3 = 0.05; alpha_4 = 0.1
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
    # Covariance in propagated state
    state_cov = np.dot(np.dot(G,prev_state_cov),np.transpose(G))+np.dot(np.dot(V,M),np.transpose(V))
    return robot_state,state_cov

'''
Performing Articulated SLAM
Pass in optional parameter for collecting debug output for all the landmarks
'''
def articulated_slam(debug_inp=True):

    # Motion probability threshold
    m_thresh = 0.75 # Choose the articulation model with greater than this threhsold's probability
    
    # Getting the map
    nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ekf_map_id = dict(); # Formulating consistent EKF mean and covariance vectors
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    
    # to get the landmarks with ids that are being seen by robot
    rob_obs_iter = landmarkmap.get_robot_observations(
        lmmap, robtraj, maxangle, maxdist, 
                                              # Do not pass visualizer to
                                              # disable visualization
                                              lmvis=None)
    rob_obs_iter = list(rob_obs_iter)
    frame_period = lmvis.frame_period
    

    # EKF parameters for filtering

    # Initially we only have the robot state
    (_, _, _, rob_state_and_input, _,_) = rob_obs_iter[0]
    slam_state =  np.array(rob_state_and_input[:3]) # \mu_{t} state at current time step
    
    # Covariance following discussion on page 317
    # Assuming that position of the robot is exactly known
    slam_cov = np.diag(np.ones(slam_state.shape[0])) # covariance at current time step
    ld_ids = [] # Landmark ids which will be used for EKF motion propagation step
    index_set = [slam_state.shape[0]] # To update only the appropriate indices of state and covariance 
    # Observation noise
    Q_obs = np.array([[5.0,0],[0,np.pi]])
    # For plotting
    obs_num = 0
    # For error estimation in robot localization
    true_robot_states = []
    slam_robot_states = []


    # Processing all the observations
    # v1.0 Need to update to v2.0 with no rs and thetas
    for fidx, (rs, thetas, ids, rob_state_and_input, ldmks,ldmk_robot_obs) in enumerate(rob_obs_iter[1:]): 
    # v2.0 Expected format
    #for fidx,(idx,rob_state_and_input, ldmks, ldmk_robot_obs) in enumerate(rob_obs_iter[1:]):    

        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state:', rob_state
        # v1.0
        #print 'Observations:', zip(rs, thetas, ids)
        # v2.0 
        print 'Observations:',ldmk_robot_obs
        
        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        
        # Following EKF steps now

        # First step is propagate : both robot state and motion parameter of any active landmark
        slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],slam_cov[0:3,0:3],robot_input)
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

                slam_cov[start_ind:end_ind,start_ind:end_ind] = ldmk_am[ld_id].prop_motion_par_cov(\
                        slam_cov[start_ind:end_ind,start_ind:end_ind])
            # end of loop over ekf propagation

        # end of if



        colors = []
        mm_probs = []
        # Collecting all the predictions made by the landmark
        ld_preds = []
        ld_ids_preds = []
        ids_list = []
    
        # Processing all the observations
        # v1.0 Update to v2.0 to exclude r and theta  
        for r, theta, id, ldmk_rob_obv in zip(rs, thetas, ids, np.dstack(ldmk_robot_obs)[0]):
        # v2.0
        # for id, ldmk_rob_obv in zip(idx,np.dstack(ldmk_robot_obs)):
           
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
                # Still need to estimate the motion class
                obs = [r, theta]
                motion_class.process_inp_data(obs, rob_state,np.vstack(ldmk_rob_obv))
                #if id == 32:
                #    print "Model Data", motion_class.am[0].model_data," Obs Num ",fidx
                #    pdb.set_trace()
                # Check if the model is estimated
                if sum(motion_class.prior>m_thresh)>0:
                    ldmk_am[id] = motion_class.am[np.where(motion_class.prior>m_thresh)[0]]
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
                ld_preds.append(lk_pred)
                ld_ids_preds.append(curr_ind)
                diff_vec = lk_pred-slam_state[0:2]
                q_val = np.dot(diff_vec,diff_vec)
                
                # v1.0
                z_pred = np.array([np.sqrt(q_val),np.arctan2(diff_vec[1],diff_vec[0])-slam_state[2]])
                # v2.0 : Need to modify to include z_pred using 3D rotation matrix obtained from the model
                #z_pred = lk_pred[:,0:3].dot(np.dstack(ldmk_rob_obv)) + lk_pred[:,3]                
                # Getting the jacobian matrix v 1.0
                H_mat = np.zeros((2,index_set[-1]))
                # v2.0 : New definition
                #H_mat = np.zeros(3,6)
                
                # w.r.t robot state
                H_mat[0,0:3] = (1.0/q_val)*np.array([-np.sqrt(q_val)*diff_vec[0],\
                        -np.sqrt(q_val)*diff_vec[1],0])
                H_mat[1,0:3] = (1.0/q_val)*np.array([diff_vec[1],-diff_vec[0],-q_val])
                # w.r.t landmark associated states
                # Differentiation w.r.t landmark x and y first
                diff_landmark = (1.0/q_val)*np.array([[np.sqrt(q_val)*diff_vec[0],\
                        np.sqrt(q_val)*diff_vec[1]],[-diff_vec[1],diff_vec[0]]])
                H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = np.dot(diff_landmark,\
                        ldmk_am[id].observation_jac(slam_state[index_set[curr_ind]:index_set[curr_ind+1]])) 

                # v2.0 Need to modify based on 3D rotation matrix jacobian: Need to find the updated theta value
                #H_mat[0,0:3] = np.array([1,0,-np.sin(theta)*ldmk_rob_obv[0] - np.cos(theta)*ldmk_rob_obv[1]])
                #H_mat[1,0:3] = np.array([0,1,(np.cos(theta)-np.sin(theta))*ldmk_rob_obv[0] - (np.cos(theta)+np.sin(theta))*ldmk_rob_obv[1]])
                #H_mat[2,0:3] = np.array([0,0,(np.sin(theta)+npcos(theta))*ldmk_rob_obv[0] + (np.cos(theta)+np.sin(theta))*ldmk_rob_obv[1]])
                #H_mat[0:3,3:6] = lk_pred[:,0:3]


                # Innovation covariance
                inno_cov = np.dot(H_mat,np.dot(slam_cov,H_mat.T))+Q_obs
                # Kalman Gain
                K_mat = np.dot(np.dot(slam_cov,H_mat.T),np.linalg.inv(inno_cov))
                
                # Updating SLAM state
                # v1.0 Need to update mode to v2.0
                slam_state = slam_state+np.dot(K_mat,(np.array([r,theta])-z_pred))
                # v2.0
                #slam_state = slam_state + np.dot(K_mat,(np.dstack(ldmk_rob_obv)-z_pred))               

 
                # Updating SLAM covariance
                slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)
            # end of if else ldmk_am[id]
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
                
        # Follow all the steps on
        print "SLAM State for robot and landmarks is",slam_state
        obs_num = obs_num+1
        print 'motion_class.priors', mm_probs
        print 'ids:', ids_list
        print 'colors:', colors
        #up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
                                  slam_cov[:2, :2], colors,obs_num)
        true_robot_states.append(rob_state)
        slam_robot_states.append(slam_state[0:3].tolist())
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

