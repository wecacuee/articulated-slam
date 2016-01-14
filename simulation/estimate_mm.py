'''
Estimate the right articulation model by observations from lots of them
The idea is to first use some measurements in batch mode to 
initialize the paramters of the articulation model and then recursively estimate
the probability of each articulation model
'''
import articulation_models_3d as am
import numpy as np
import scipy.stats as sp
import pdb

# Robot bearing and range to x,y in cartesian frame given the robot state
def bearing_to_cartesian(obs,robot_state,gen_obv):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    
    # v1.0 Observation is of range and bearing angle
    r = obs[0]; phi = obs[1]
    
    return np.array([x+r*np.cos(theta+phi),y+r*np.sin(theta+phi)])

# v2.0 function used instead of bearing_to_cartesian
def robot_to_world(robot_state,gen_obv):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    
    # v2.0 Using inverse rotation matrix to retrieve landmark world frame coordinates
    R = np.array([[np.cos(theta), -np.sin(theta),0],
                [np.sin(theta), np.cos(theta),0],
                [0,0,1]])
    
    # v2.0 
    return np.reshape(R.T.dot(gen_obv) + np.array([[x],[y],[0]]),[1,3])[0]

# x,y in cartesian to robot bearing and range
def cartesian_to_bearing(obs,robot_state):
    # Robot state consists of (x,y,\theta) where \theta is heading angle
    x = robot_state[0]; y= robot_state[1]; theta = robot_state[2]
    # Observation is of x and y
    m_x = obs[0]; m_y = obs[1]
    return np.array([np.sqrt((m_x-x)**2+(m_y-y)**2),np.arctan2(m_y-y,m_x-x)])

class Estimate_Mm:
    # Constructor for class
    def __init__(self):
        # Initialize motion models
        # Noise in all the noise models for motion propagation
        # self.noise_motion = np.diag([0.01,0.01])
        # self.noise_obs = np.diag([0.01,0.01])
        self.noise_motion = np.diag([0.01,0.01]) * 10
        self.noise_obs = np.diag([0.01,0.01,0.01]) * 100
        # To keep track of the number of samples that have been processed
        self.num_data = 0
        # Refer to landmarks_motion_models.py for more details
        self.am = [am.Revolute_Landmark(),
                am.Prismatic_Landmark(),
                am.Static_Landmark()]
        # To store the means of the state
        self.means = [np.array((2,))]*len(self.am)
        # To store the covariance of all the states
        self.covs = [np.array((2,2))]*len(self.am)
        # Prior probabilities of the motion models
        self.prior = (1.0/len(self.am))*np.ones((len(self.am),))
        
        # Minimum number of observations in order to estimate all the model
        if len(self.am)>0:
            self.min_samples = max(3,max(am.min_data_samples \
                    for am in self.am))


    # To process input data -- input data is from robot sensor [r,\theta]
    # robot_state is the position of the robot from where this data was sensed
    def process_inp_data(self,inp_data_bearing,robot_state,ldmk_rob_obv,init_pt):
        self.num_data = self.num_data+1
        lk_prob = np.zeros(len(self.am))
        residual = list()
        inno_covariances = list()
        # All the motion models work in x,y but we get bearing and range from sensor
        # v2.0 Modified bearing_to_catersian function to use only ldmk_rob_obv variable
        inp_data = robot_to_world(robot_state,ldmk_rob_obv)

        # Pass this data to all the models
        for i in range(len(self.am)):
            if (self.num_data<self.min_samples):
                self.am[i].process_inp_data(inp_data)
            # State mean and covariances can be updated
            elif (self.num_data==self.min_samples):
                self.am[i].process_inp_data(inp_data)
                self.means[i] = self.am[i].motion_pars[0].state
                self.covs[i] = np.diag(np.tile(self.am[i].noise_cov,(self.means[i].shape[0],)))
                # Process samples if its more the minimum required number of samples
            else :
                # Perform EKF for estimating which model to use
                # Step 1: Propagate State
                # All these motion models only have one motion parameter
                self.means[i] = self.am[i].predict_motion_pars(self.means[i])
                # Step 2: Propagate Covariance
                model_lin_mat = self.am[i].model_linear_matrix()
                self.covs[i] = model_lin_mat.dot(self.covs[i]).dot(model_lin_mat.T)+\
                                np.diag(np.tile(self.am[i].noise_cov,(self.means[i].shape[0],)))
                # Step 3.0: Compute Innovation Covariance
                H_t = self.am[i].observation_jac(self.means[i],init_pt)
                inno_cov = H_t.dot(self.covs[i]).dot(H_t.T)+self.noise_obs

                # Step 3: Compute Kalman Gain
                K_t = np.dot(np.dot(self.covs[i],np.transpose(H_t)),np.linalg.inv(inno_cov))
                # Step 4: Update State
                residual.append( inp_data -
                        self.am[i].predict_model(self.means[i]))
                #print "Prediction is ", self.am[i].predict_model(self.means[i])," mean is ", \
                #        self.means[i]," input data is ",inp_data
                self.means[i] = self.means[i]+np.dot(K_t,residual[-1])
                # Step 5: Update State Covariance
                self.covs[i] = np.dot(np.identity(self.means[i].shape[0])\
                        -np.dot(K_t,H_t),self.covs[i])

                # Update the model contribution -- 11.6.2-2 of Estimation with Applications to
                # tracking and navigation

                # Likelihood function and probability
                lk_prob[i] = sp.multivariate_normal.pdf(residual[-1],mean = np.array([0,0,0]),
                        cov = inno_cov)
                inno_covariances.append(np.linalg.det(inno_cov))

        if (self.num_data>self.min_samples):
            for i in range(len(self.am)):
                self.prior[i] = self.prior[i]*lk_prob[i]

        # Updating the model probabilities to sum to 1
        self.prior = self.prior/np.sum(self.prior)
        return self.prior


if __name__=="__main__":
    robot_state = np.array([0,0,0])
    motion_class = Estimate_Mm()
    # Lets generate data from a revolute joint centered at (2,2), radius 1, moving at pi/6 pace
    r = 1;x_0 = 2;y_0 = 2;w = np.pi/6
    for i in range(30):
        # Revolute
        curr_obs = np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0])

        # Static
        #curr_obs = np.array([x_0,y_0])

        # Prismatic - starting from x_0,y_0 ,slope of line at w
        #curr_obs = np.array([x_0-i*r*np.cos(w),y_0+i*r*np.sin(w)])
        motion_class.process_inp_data(cartesian_to_bearing(curr_obs,robot_state),robot_state)
        print "Rev: ",motion_class.prior[0],"Pris: ",motion_class.prior[1],"Static: ",motion_class.prior[2]
