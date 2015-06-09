'''
Implement finite order temporal models of motion
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import pdb

class temporal_model:
    # Constructor for the class
    def __init__(self,init_state = np.zeros(2),_dt=1):
        # Define the order of the temporal model
        # Order 1 is the constant velocity model
        self.order = len(init_state)
        # Define the initial state of a finite order temporal model
        self.state = init_state
        # Noise characteristics for the last state derivative, for example for a constant velocity model, we will assume acceleration as noise
        self.noise_sigma = 0.1/len(init_state)
        self.noise_mu = 0 # Assuming zero mean noise
        # Minimum number of samples required
        self.min_data_samples = 2*int(self.order/2.0)+5 # Ensure odd
        # To store the latest model data
        self.model_data = np.zeros(self.min_data_samples)
        self.num_data = 0
        # Time difference between two consecutive measurements
        self._dt = _dt
        # Parameters of the model, its all about getting the last state of the
        # variable right
        self.model_pars = np.zeros(len(init_state))
        # Sigma for gaussian smoothing to be used for derivative computation
        self.gauss_sigma = 1

    # Forward propagation model
    def forward_motion(self,dt):
        # Generate a sample for last state by sampling from gaussian with given noise variance
        noise_input = self.noise_mu+self.noise_sigma*np.random.randn()
        # Dummy variable for state prediction
        pred_state = np.zeros(len(self.state))
        for i in range(self.order):
            for j in range(i,self.order):
                pred_state[i] = pred_state[i]+((dt**(j-i))/math.factorial(j-i))*self.state[j]
                # Adding noise to the final state
                pred_state[i] = pred_state[i]+noise_input
        # Assign the predicted state to current state
        self.state = pred_state

    # To find the finite order derivative of a sequence
    def fit_model(self):
        # Denominator
        den = self._dt**(self.order-1)
        # Convolving with gaussian to find derivative
        gf = ndimage.gaussian_filter1d(self.model_data,
                                       sigma=self.gauss_sigma,
                                       order=self.order-1,mode='wrap') / den
        # Getting the last state parameter from the array
        # Assuming that the motion is that order continuous
        self.model_pars[-1] = gf[int(len(gf)/2.0)]
        # First state is the position which is the state itself
        self.model_pars[0] = self.model_data[-1]
        # Getting the rest of the states
        for i in range(self.order-1,1,-1):
            # Denominator
            den = self._dt**i
            # Removing the contribution of last computed derivative state
            for j in range(1,len(self.model_data)):
                self.model_data[j] = self.model_data[j]-\
                        self.model_pars[i]*((self._dt*j)**(i)/(math.factorial(i)*1.0))
            # Convolving with gaussian to find derivative
            gf =ndimage.gaussian_filter1d(self.model_data,
                                          sigma=self.gauss_sigma,
                                          order=i-1,mode='reflect') / den
            # Getting the current state
            self.model_pars[i-1] = gf[-1-i]
        # Print estimated model parameters
        print "Estimated Model parameters are",self.model_pars

    # Define the linear model for covariance propagation
    def model_linear_matrix(self):
        dt = self._dt # Time step
        model_mat = np.zeros([self.order,self.order])
        for i in range(self.order):
            for j in range(i,self.order):
                model_mat[i][j] = ((dt**(j-i))/math.factorial(j-i))
        return model_mat

    # Main input data processing code
    def process_inp_data(self,inp_data):
        # Main function to receive input data and call appropriate functions
        self.num_data = self.num_data+1
        if self.num_data<self.min_data_samples:
            # Add data to model data
            self.model_data[self.num_data-1] = inp_data.copy()
        elif self.num_data==self.min_data_samples:
            # Add data and call fit_model function
            self.model_data[self.num_data-1] = inp_data.copy()
            self.fit_model()
        else:
            # Update the stored data as well
            self.model_data[1:] = self.model_data[:-1]
            self.model_data[-1] = inp_data
            # Doing an EKF kind of algorithm to test maneuver detection 
            # and changing state parameters based on that
            
            # Step 1:  
 

# Define the observation model for getting the kalman filtering step
def observation_model(robot_state):
    # Right now we are just estimating position information
    return robot_state[0]

# Define the jacobian model for getting EKF results
def observation_jac(robot_state):
    # Since we are only using the position level measurements
    jac = np.zeros(len(robot_state))
    jac[0] = 1
    return jac



if __name__=="__main__":
    init_state = np.array([10,4,2,1])
    const_vel = temporal_model(init_state)
    dt = 1.0 # Time steps of dt sec
    # Simulate for t_time seconds
    t_time = 10
    position = np.zeros(int(t_time/dt))
    for i in range(int(t_time/dt)):
        const_vel.process_inp_data(const_vel.state[0]) # Using this as input
        position[i] = const_vel.state[0]
        const_vel.forward_motion(dt) # First generating data from motion

    plt.plot(position)
    plt.show()



