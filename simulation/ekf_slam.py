'''
Perform the actual SLAM on the landmarks and robot positions
'''
import landmarkmap
import cv2
import numpy as np
import scipy.stats as sp
import pdb
import utils_plot as up
import scipy.linalg
import matplotlib.pyplot as plt



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


# x,y in cartesian to robot bearing and range
def cartesian_to_bearing(obs,robot_state):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Observation is of x and y
    m_x = obs[0]; m_y = obs[1]
    return np.array([np.sqrt((m_x-x)**2+(m_y-y)**2),np.arctan2(m_y-y,m_x-x)])

def threeptmap():
    nframes = 100
    map_conf = [# static
                dict(ldmks=np.array([[0, 0]]).T,
                     inittheta=0,
                # Where do we obtain the x and y from?   
                     initpos=[x, y],
                     deltheta=0,
                     delpos=[0,0]) 
                for x,y in zip([10] * 10 + range(10, 191, 20) + [190]*10 +
                               range(10, 191, 20),
                               range(10, 191, 20) + [190]*10 + 
                               range(10, 191, 20) + [10] * 10
                              )
               ]+ [# prismatic
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
    # Probabilistic Robotics book
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
Performing EKF SLAM
Pass in optional parameter for collecting debug output for all the landmarks
'''
def slam(debug_inp=True):

    # Getting the map
    nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_am = dict(); # To keep a tab on the landmarks that have been previously seen

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
    (_, _, _, rob_state_and_input, _) = rob_obs_iter[0]
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
    # Initial covariance for landmarks (x,y) position
    initial_cov = np.array([[100,0],[0,100]])
    # For error estimation in robot localization
    true_robot_states = []
    slam_robot_states = []

    # Processing all the observations
    # We need to skip the first observation because it was used to initialize SLAM State
    for fidx, (rs, thetas, ids, rob_state_and_input, ldmks) in enumerate(rob_obs_iter[1:]): 
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot true state:', rob_state
        print 'Observations:', zip(rs, thetas, ids)
        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        
        # Following EKF steps now

        # First step is propagate : both robot state and motion parameter of any active landmark
        slam_state[0:3],slam_cov[0:3,0:3]=robot_motion_prop(slam_state[0:3],slam_cov[0:3,0:3],robot_input)

        colors = []
        mm_probs = []
        # Collecting all the predictions made by the landmark
        ids_list = []
        # Processing all the observations
        for r, theta, id in zip(rs, thetas, ids):
            # Observation corresponding to current landmark is
            obs = np.array([r, theta])
            '''
            Step 1: Process Observations to first determine the motion model of each landmark
            During this step we will use robot's internal odometry to get the best position of 
            external landmark as we can and try to get enough data to estimate landmark's motion
            model. 
            '''
            # Setting default none value for the current landmark observation
            ldmk_am.setdefault(id,None)
            # For each landmark id, we want to check if the landmark has been previously seen
            if ldmk_am[id] is None:
                # Assign a static landmark id
                ldmk_am[id] = 2
                ld_ids.append(id)
                # Getting the current state to be added to the SLAM state (x,y) position of landmark
                curr_ld_state = bearing_to_cartesian(obs,rob_state)
                curr_ld_cov = initial_cov 
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
                lk_pred = bearing_to_cartesian(obs,slam_state[0:3])
                diff_vec = lk_pred-slam_state[0:2]
                q_val = np.dot(diff_vec,diff_vec)
                z_pred = np.array([np.sqrt(q_val),np.arctan2(diff_vec[1],diff_vec[0])-slam_state[2]])
                # Getting the jacobian matrix 
                H_mat = np.zeros((2,index_set[-1]))
                # w.r.t robot state
                H_mat[0,0:3] = (1.0/q_val)*np.array([-np.sqrt(q_val)*diff_vec[0],\
                        -np.sqrt(q_val)*diff_vec[1],0])
                H_mat[1,0:3] = (1.0/q_val)*np.array([diff_vec[1],-diff_vec[0],-q_val])
                # w.r.t landmark associated states
                # Differentiation w.r.t landmark x and y first
                H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = (1.0/q_val)*np.array(\
                        [[np.sqrt(q_val)*diff_vec[0],np.sqrt(q_val)*diff_vec[1]],\
                        [-diff_vec[1],diff_vec[0]]])

                # Innovation covariance
                inno_cov = np.dot(H_mat,np.dot(slam_cov,H_mat.T))+Q_obs
                # Kalman Gain
                K_mat = np.dot(np.dot(slam_cov,H_mat.T),np.linalg.inv(inno_cov))
                # Updating SLAM state
                slam_state = slam_state+np.dot(K_mat,(obs-z_pred))
                # Updating SLAM covariance
                slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)
            # end of if else ldmk_am[id]

            p1, p2, p3 = (0,0,1) # We are assuming everything to be static
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
        #up.slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds)
        #visualize_ldmks_robot_cov(lmvis, ldmks, robview, slam_state[:2],
        #                          slam_cov[:2, :2], colors)
        true_robot_states.append(rob_state)
        slam_robot_states.append(slam_state[0:3].tolist())
    # end of loop over frames
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

