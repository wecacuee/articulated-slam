'''
Performs the main goal of this project, being able to use articulation
inside SLAM, by first
- Getting few observations and estimating the articulated models
- Using the motion parameters of those models to propagate the landmarks and the robot
- Using observations to update the EKF state (consisting of robot pose and motion parameters)

- Code is similar to threeptmap.py
'''

import numpy as np
import landmarkmap
import cv2
import estimate_mm as mm # To figure out the right motion model
import pdb

def threeptmap():
    nframes = 20
    map_conf = [dict(ldmks=np.array([[10, 10]]).T,
                     inittheta=0,
                     initpos=[80, 10],
                     deltheta=0,
                     delpos=[0,0]),
                # prismatic
                dict(ldmks=np.array([[10,10]]).T,
                     inittheta=0,
                     initpos=[20,10],
                     deltheta=0,
                     delpos=[5,0]),
                # revolute
                dict(ldmks=np.array([[0,20]]).T,
                     inittheta=np.pi,
                     initpos=[50,40],
                     deltheta=-10*np.pi/180,
                     delpos=[0,0])
               ]
    lmmap = landmarkmap.map_from_conf(map_conf, nframes)
    lmvis = landmarkmap.LandmarksVisualizer([0,0], [100, 100], frame_period=-1,
                                         scale=3)
    robtraj = landmarkmap.robot_trajectory(np.array([[10, 90], [40,60]]),
                                           [nframes], np.pi/10)
    # angle on both sides of robot dir
    maxangle = 45*np.pi/180
    # max distance in pixels
    maxdist = 120

    return nframes, lmmap, lmvis, robtraj, maxangle, maxdist


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
    for fidx, (rs, thetas, ids, rob_state, ldmks) in enumerate(rob_obs_iter): 
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
Inputs are: Previous robot state, actual robot input (translational and rotational component),
covarinace in previous state, and time interval
'''
def robot_motion_prop(prev_state,robot_input,prev_state_cov,delta_t):
    # Robot input is [v,w]^T where v is the linear velocity and w is the rotational component
    v = robot_input[0];w=robot_input[1];
    # Robot state is [x,y,\theta]^T, where x,y is the position and \theta is the orientation
    x = prev_state[0]; y=prev_state[1];theta = prev_state[2]
    robot_state = np.zeros(3)
    # Setting noise parameters, following Page 210 Chapter 7, Mobile Robot Localization of 
    # Probabilistic Robotics book
    alpha_1 = 0.1; alpha_2=0.05; alpha_3 = 0.05; alpha_4 = 0.1
    # M for transferring noise from input space to state-space
    M = np.array([[(alpha_1*v^2)+(alpha_2*w^2),0],[0,(alpha_3*v^2)+(alpha_4*w^2)]])
    
    # Check if rotational velocity is close to 0
    if (abs(w)<1e-4):
        robot_state[0] = x+(v*delta_t*np.cos(theta))        
        robot_state[1] = y+(v*delta_t*np.sin(theta))        
        robot_state[2] = theta
        # Derivative of forward dynamics model w.r.t previous robot state
        G = np.array([[1,0,-v*delta_t*np.cos(theta)],\
                [0,1,v*delta_t*np.sin(theta)],\
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
                (v*(np.sin(theta)-sin(theta+w*delta_t)))/(w**2)+((v*np.cos(theta+w*delta_t)*delta_t)/w)],\
                [(np.cos(theta)-np.cos(theta+w*delta_t))/w,\
                (-v*(np.cos(theta)-np.cos(theta+w*delta_t)))/(w**2)+((v*np.sin(theta+w*delta_t)*delta_t)/w)],\
                [0,delta_t]])
    # Covariance in propagated state
    state_cov = np.dot(np.dot(G,prev_state_cov),np.transpose(G))+np.dot(np.dot(V,M),np.transpose(V))
    return robot_state,state_cov


def articulated_slam():
    # Motion probability threshold
    m_thresh = 0.8 # Choose the articulation model with greater than this threhsold's probability
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
    frame_period = lmvis.frame_period
    # EKF related parameters
    prev_state = [] # \mu_{t-1} state at previous time step
    prev_cov = [] # \Sigma_{t-1} Covariance at previous time step

    state = [] # \mu_{t} state at current time step
    cov = [] # covariance at current time step


    # Processing all the observations
    for fidx, (rs, thetas, ids, rob_state, ldmks) in enumerate(rob_obs_iter): 
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state:', rob_state
        print 'Observations:', zip(rs, thetas)
        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        colors = []
        mm_probs = []
        ld_ids = [] # Landmark ids which will be used for EKF in current iteration
        for r, theta, id in zip(rs, thetas, ids):
            motion_class = ldmk_estimater.setdefault(id, mm.Estimate_Mm())
            # For storing the chosen articulated model
            chosen_am = ldmk_am.setdefault(id,None)
            '''
            Step 1: Process Observations to first determine the motion model of each landmark
            During this step we will use robot's internal odometry to get the best position of 
            external landmark as we can and try to get enough data to estimate landmark's motion
            model. 
            '''
            # EKF has to be done for this
            if ldmk_am[id] is not None:
                # Have to pass this landmark to EKF estimator
                ld_ids.append(id)
            else:
                # Still need to estimate the motion class
                obs = [r, theta]
                motion_class.process_inp_data(obs, rob_state)
                mm_probs.append(motion_class.prior)
                # Check if the model is estimated
                if sum(motion_class.prior>m_thresh)>0:
                    ldmk_am[id] = motion_class.am[np.where(motion_class.prior>m_thresh)[0]]
                    ekf_map_id[len(ekf_map_id)+1] = id
        
        # Following EKF steps now
        if len(ld_ids)>0:
            '''
            Step 2: When any landmark's motion model is estimated, we start doing EKF SLAM with 
            state as robot's pose and motion parameter of each of the landmarks currently estimated
            '''

        print 'motion_class.priors', mm_probs


if __name__ == '__main__':
    # For reproducing the similar results with newer version of code
    mapping_example()
    #articulated_slam()



