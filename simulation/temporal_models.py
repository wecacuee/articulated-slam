'''
Implement finite order temporal models of motion
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.stats as sp
import pdb
import xlrd

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
        self.min_data_samples = max(5,2*int(self.order/2.0)+1) # Ensure odd
        # To store the latest model data
        self.model_data = np.zeros(self.min_data_samples)
        self.num_data = 0
        # Time difference between two consecutive measurements
        self._dt = _dt
        # Parameters of the model, its all about getting the last state of the
        # variable right
        self.model_pars = np.zeros(len(init_state))
        # Sigma for Gaussian smoothing to be used for derivative computation
        self.gauss_sigma = max(0.6,0.13) # Length of the kernel is 2*lw+1 where lw is int(4*sigma+0.5)
        # Because of the above restriction, gauss_sigma should atleast be 0.125 for smoothing
        # Noise in the motion propagation step
        self.noise_motion = np.diag(0.01*np.ones(self.order))
        # Noise in the observation step
        self.noise_obs = 1.0
        # State covariance
        self.cov = np.diag(1*np.ones(self.order))
        # Initial likelihood probability of the model
        self.init_lk_prob = None
        # Theshold for fraction by which the probability needs to decrease
        self.prob_thresh = 0.1


    # Forward propagation model
    def forward_motion(self,state):
        # Time step
        dt = self._dt
        # Generate a sample for last state by sampling from Gaussian with given noise variance
        # Add noise only if we are trying to simulate the true generation process
        # However, usually we are using this within a filtering framework, where only the deterministic part needs to be propagated
        noise_input = 0.0
        # noise_input = self.noise_mu+self.noise_sigma*np.random.randn()
        # Dummy variable for state prediction
        pred_state = np.zeros(len(self.state))
        for i in range(self.order):
            for j in range(i,self.order):
                pred_state[i] = pred_state[i]+((dt**(j-i))/math.factorial(j-i))*state[j]
                # Adding noise to the final state
                pred_state[i] = pred_state[i]+noise_input
        return pred_state

    # To find the finite order derivative of a sequence
    def fit_model(self):

        '''
        Right now using convolution of time series with derivatives of gaussian to estimate
        temporal derivatives but can use other better methods such as
        https://hal.inria.fr/inria-00319240/document
        '''
        # Denominator
        den = self._dt**(self.order-1)
        # Convolving with gaussian to find derivative
        gf = ndimage.gaussian_filter1d(self.model_data,
                                       sigma=self.gauss_sigma,
                                       order=self.order-1,mode='wrap') / den
        print "2nd order",gf[int(len(gf)/2.0)]
        # Getting the last state parameter from the array
        # Assuming that the motion is that order continuous
        self.model_pars[-1] = gf[int(len(gf)/2.0)]
        # First state is the position which is the state itself
        self.model_pars[0] = self.model_data[-1]
        print self.model_data
        gf = ndimage.gaussian_filter1d(self.model_data,
                                       sigma=self.gauss_sigma,
                                       order=1,mode='wrap') / den
        print "1st order",gf[int(len(gf)/2.0)]

        # Getting the rest of the states
        for i in range(self.order-1,1,-1):
            # Denominator
            den = self._dt**i
            # Removing the contribution of last computed derivative state
            for j in range(1,len(self.model_data)):
                self.model_data[j] = self.model_data[j]-\
                        self.model_pars[i]*((self._dt*j)**(i)/(math.factorial(i)*1.0))
            # Convolving with Gaussian to find derivative
            gf =ndimage.gaussian_filter1d(self.model_data,
                                          sigma=self.gauss_sigma,
                                          order=i-1,mode='reflect') / den
            # Getting the current state
            self.model_pars[i-1] = gf[-1-i]
            print "Gaussian:",gf
        # Print estimated model parameters
        print "Estimated temporal model parameters are",self.model_pars
        #pdb.set_trace()

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
        lk_prob = 0
        if self.num_data<self.min_data_samples:
            # Add data to model data
            self.model_data[self.num_data-1] = inp_data.copy()
        elif self.num_data==self.min_data_samples:
            # Add data and call fit_model function
            self.model_data[self.num_data-1] = inp_data.copy()
            self.fit_model()
            # Update the state based on estimated model paramters
            self.state = self.model_pars
        else:
            # Update the stored data as well
            self.model_data[1:] = self.model_data[:-1]
            self.model_data[-1] = inp_data
            # Doing an EKF kind of algorithm to test maneuver detection 
            # and changing state parameters based on that
            # Step 1: Propagate State
            self.state = self.forward_motion(self.state)
            
            # Step 2: Propagate Covariance
            model_lin_mat = self.model_linear_matrix()
            self.cov = model_lin_mat.dot(self.cov).dot(model_lin_mat.T)+self.noise_motion

            # Step 3.0: Compute Innovation Covariance
            H_t = observation_jac(self.state)
            inno_cov = H_t.dot(self.cov).dot(H_t.T)+self.noise_obs

            # Step 3: Compute Kalman Gain
            K_t = np.dot(self.cov,np.transpose(H_t))*(1.0/inno_cov)
            # Step 4: Update State
            residual = inp_data-observation_model(self.state)
            self.state = self.state+np.dot(K_t,residual)
            # Step 5: Update State Covariance
            # Special treatment because K_t is an array but we need matrix multiplication
            self.cov = np.dot(np.identity(self.order)-np.dot(K_t[np.newaxis].T,H_t[np.newaxis]),self.cov)

            # Update the model contribution -- 11.6.2-2 of Estimation with Applications to
            # tracking and navigation

            # Likelihood function and probability
            lk_prob = sp.multivariate_normal.pdf(residual,mean = np.array([0,0]),
                    cov = inno_cov)
            #print lk_prob,self.state,inp_data
            # Set the model's initial probability
            if self.init_lk_prob is None:
                self.init_lk_prob = lk_prob
            else:
                if (lk_prob/self.init_lk_prob < self.prob_thresh):
                    print "The model is no longer valid"
                    pdb.set_trace()

        return lk_prob 

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
    init_state = np.array([0])
    dt = 1.0 # Time steps of dt sec
    const_vel = temporal_model(init_state,dt)
    # Reading the data from the excel file
    wb = xlrd.open_workbook('../temp/output_angles_ground_truth.xls')
    sh = wb.sheet_by_index(0)
    num_points = 30
    inp_data = np.zeros(num_points+10)
    lk_prob = np.zeros(num_points+10)
    out_state = np.zeros(num_points+10)
    for i in range(1,num_points+1):
        inp_data[i-1] = sh.cell(0,i).value
    inp_data[-10:] = inp_data[-11]
    print inp_data
    num_points = num_points+10
    # Passing this data to the algorithm
    for i in range(inp_data.shape[0]):
        lk_prob[i] = const_vel.process_inp_data(inp_data[i])
        out_state[i] = const_vel.state[0]

    plt.subplot(2, 1, 1)
    inp = plt.plot(inp_data,'b+',label='Input Data',linewidth=2.0)
    out = plt.plot(np.arange(const_vel.min_data_samples,num_points),
                   out_state[const_vel.min_data_samples:],
                   'r', label='State Output', linewidth=2.0)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc='upper left')

    plt.subplot(2,1,2)
    out = plt.plot(np.arange(const_vel.min_data_samples,num_points),
                   lk_prob[const_vel.min_data_samples:],
                   'r', linewidth=2.0)
    plt.xlim([0,num_points])
    plt.xlabel("Time")
    plt.ylabel("Prob")
    plt.show()

