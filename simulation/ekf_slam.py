'''
Perform the actual SLAM on the landmarks and robot positions
'''
import landmarks_motion_models as mm
import numpy as np
import scipy.stats as sp
import pdb

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

'''
Estimate the right motion model by observations from lots of them
The idea is to first use some measurements in batch mode to 
initialize the paramters of the motion model and then recursively estimate
the probability of each motion model
'''
class Estimate_Mm:
    # Constructor for class
    def __init__(self):
        # Initialize motion models
        # Noise in all the noise models for motion propagation
        # self.noise_motion = np.diag([0.01,0.01])
        # self.noise_obs = np.diag([0.01,0.01])
        self.noise_motion = np.diag([0.01,0.01]) * 100
        self.noise_obs = np.diag([0.01,0.01]) * 100
        # Minimum number of observations in order to estimate all the model
        self.min_samples = 3
        # To keep track of the number of samples that have been processed
        self.num_data = 0
        # Refer to landmarks_motion_models.py for more details
        self.mm = [mm.Revolute_Landmark(3,self.noise_motion),
                mm.Prismatic_Landmark(2,self.noise_motion),
                mm.Static_Landmark(1,self.noise_motion)]
        # To store the means of the state
        self.means = [np.array((2,))]*len(self.mm)
        # To store the covariance of all the states
        self.covs = [np.array((2,2))]*len(self.mm)
        # Prior probabilities of the motion models
        self.prior = (1.0/len(self.mm))*np.ones((len(self.mm),))


    # To process input data -- input data is from robot sensor [r,\theta]
    # robot_state is the position of the robot from where this data was sensed
    def process_inp_data(self,inp_data,robot_state):
        self.num_data = self.num_data+1
        lk_prob = np.zeros(len(self.mm))
        residual = list()
        inno_covariances = list()
        # Pass this data to all the models
        for i in range(len(self.mm)):
            # All the motion models work in x,y but we get bearing and range from sensor
            self.mm[i].process_inp_data(bearing_to_cartesian(inp_data,robot_state))
            # State mean and covariances can be updated
            if (self.num_data==self.min_samples):
                self.means[i] = self.mm[i].model_data[-1,:].copy()
                self.covs[i] = self.mm[i].noise_cov
                # Process samples if its more the minimum required number of samples
            elif (self.num_data>self.min_samples):
                # Perform EKF for estimating which model to use
                # Step 1: Propagate State
                self.means[i] = self.mm[i].predict_model(self.means[i])
                # Step 2: Propagate Covariance

                model_lin_mat = self.mm[i].model_linear_matrix()
                self.covs[i] = model_lin_mat.dot(self.covs[i]).dot(model_lin_mat.T)+self.noise_motion

                # Step 3.0: Compute Innovation Covariance
                H_t = observation_jac(robot_state,self.means[i])
                inno_cov = H_t.dot(self.covs[i]).dot(H_t.T)+self.noise_obs

                # Step 3: Compute Kalman Gain
                K_t = np.dot(np.dot(self.covs[i],np.transpose(H_t)),np.linalg.inv(inno_cov))
                # Step 4: Update State
                residual.append( inp_data -
                        observation_model(robot_state,self.means[i]) )
                self.means[i] = self.means[i]+np.dot(K_t,residual[-1])
                # Step 5: Update State Covariance
                self.covs[i] = np.dot(np.identity(2)-np.dot(K_t,H_t),self.covs[i])

                # Update the model contribution -- 11.6.2-2 of Estimation with Applications to
                # tracking and navigation

                # Likelihood function and probability
                lk_prob[i] = sp.multivariate_normal.pdf(residual[-1],mean = np.array([0,0]),
                        cov = inno_cov)
                inno_covariances.append(np.linalg.det(inno_cov))

        if (self.num_data>self.min_samples):
            for i in range(len(self.mm)):
                self.prior[i] = self.prior[i]*lk_prob[i]

        # Updating the model probabilities to sum to 1
        self.prior = self.prior/np.sum(self.prior)
        return self.prior



if __name__=="__main__":
    robot_state = np.array([0,0,0])
    motion_class = Estimate_Mm()
    # Lets generate data from a revolute joint centered at (2,2), radius 1, moving at pi/6 pace
    r = 1;x_0 = 2;y_0 = 2;w = np.pi/6
    for i in range(10):
        # Revolute
        #curr_obs = np.array([r*np.cos(i*w)+x_0,r*np.sin(i*w)+y_0])

        # Static
        # curr_obs = np.array([x_0,y_0])

        # Prismatic - starting from x_0,y_0 ,slope of line at w
        curr_obs = np.array([x_0+i*r*np.cos(w),y_0+i*r*np.sin(w)])
        motion_class.process_inp_data(cartesian_to_bearing(curr_obs,robot_state),robot_state)
        print "Rev: ",motion_class.prior[0],"Pris: ",motion_class.prior[1],"Static: ",motion_class.prior[2]

