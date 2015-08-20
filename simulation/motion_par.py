'''
All articulated motion in the world can be represented as 
X(t) = f(C,m(t))
where X(t) is the observed motion of an object
C is the configuration space (doors open about an axis)
m(t) is the motion parameter (Ex: length of prismatic joint, 
angle of door)

This class models the motion parameter of articulated joints.
In particular, it implement finite order temporal models of motion
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.stats as sp
import pdb
import xlrd
import warnings

##
# @brief Main class for motion parameter of an articulated model
class motion_par:
    # @brief Constructor for the class
    #
    # @param _order Order of the motion model, minimum 1 for static model
    # @param _dt Time step between consecutive observations
    #
    # @return none
    def __init__(self,_order=2,_dt=1):
        # Define the order of the temporal model
        # Currently this module uses gaussian filtering which as implemented by scipy
        # only deals with derivatives upto order 3
        if _order<4:
            self.order = _order
        else:
            warnings.warn("Can not set order more than 3")
            self.order = 3
        # Define the initial state of a finite order temporal model
        self.state = None
        # Noise characteristics for the last state derivative, for example for
        # a constant velocity model, we will assume acceleration as noise
        self.noise_sigma = 0.1/self.order
        self.noise_mu = 0 # Assuming zero mean noise
        # Minimum number of samples required
        self.min_data_samples = max(7,4*int(self.order/2.0)+1) # Ensure odd
        # To store the latest model data
        self.model_data = np.zeros(self.min_data_samples)
        self.num_data = 0
        # Time difference between two consecutive measurements
        self._dt = _dt
        # Parameters of the model, its all about getting the last state of the
        # variable right
        self.model_pars = np.zeros(self.order)
        # Sigma for Gaussian smoothing to be used for derivative computation
        self.gauss_sigma = 0.6 # Length of the kernel is 2*lw+1 where lw is int(4*sigma+0.5)
        # Because of the above restriction, gauss_sigma should atleast be 0.125 for smoothing


    ##
    # @brief Forward propagation model
    #
    # @param state Current state of the motion parameter
    #
    # @return Prediction of the state at next time step 
    def propagate(self,state):
        # Time step
        dt = self._dt
        # Generate a sample for last state by sampling from Gaussian with given noise variance
        # Add noise only if we are trying to simulate the true generation process
        # However, usually we are using this within a filtering framework, 
        # where only the deterministic part needs to be propagated
        noise_input = 0.0
        # noise_input = self.noise_mu+self.noise_sigma*np.random.randn()

        # Asserting that we have a valid current state to proceed from
        assert self.state is not None
        # Dummy variable for state prediction
        pred_state = np.zeros(len(self.state))
        # Logic for writing discrete state propagation. For example
        '''
        [x(k);dx(k);ddx(k)] = [1,dt,dt^2/2;0,1,dt;0,0,1][x(k-1);dx(k-1);ddx(k-1)]
        '''
        for i in range(self.order):
            for j in range(i,self.order):
                pred_state[i] = pred_state[i]+((dt**(j-i))/math.factorial(j-i))*state[j]
                # Adding noise to the final state
                pred_state[i] = pred_state[i]+noise_input
        return pred_state

    ##
    # @brief To find the finite order derivative of a sequence
    #
    # @return None 
    def estimate_params(self):

        '''
        Right now using convolution of time series with derivatives of gaussian to estimate
        temporal derivatives but can use other better methods such as
        https://hal.inria.fr/inria-00319240/document
        '''
        # Getting the last state parameter from the array
        # Assuming that the motion is that order continuous
        # First state is the position which is the state itself
        self.model_pars[0] = self.model_data[-1]

        print "Model_data is ",self.model_data

        # Getting the rest of the states
        for i in range(self.order,1,-1):
            # Denominator
            den = self._dt**i
            # Convolving with Gaussian to find derivative
            gf =ndimage.gaussian_filter1d(self.model_data,
                                          sigma=self.gauss_sigma,
                                          order=i-1,mode='wrap')/ den
            print (i-1),"th order derivative is ",gf[int(len(gf)/2.0)]
            # Getting the current derivative
            self.model_pars[i-1] = gf[int(len(gf)/2.0)]
        # Print estimated model parameters
        print "Estimated temporal model parameters are",self.model_pars

    ##
    # @brief Define the linear model for covariance propagation
    #
    # @return Model Matrix of the motion parameter class 
    def model_linear_matrix(self):
        # This is the G matrix as used in EKF models
        dt = self._dt # Time step
        model_mat = np.zeros([self.order,self.order])
        for i in range(self.order):
            for j in range(i,self.order):
                model_mat[i][j] = ((dt**(j-i))/math.factorial(j-i))
        return model_mat

    ##
    # @brief  Main input data processing 
    #
    # @param inp_data Input data for the current parameter setup
    #
    # @return 
    def process_inp_data(self,inp_data):
        # Main function to receive input data and call appropriate functions
        self.num_data = self.num_data+1
        if self.num_data<self.min_data_samples:
            # Add data to model data
            self.model_data[self.num_data-1] = inp_data.copy()
        elif self.num_data==self.min_data_samples:
            # Add data and call estimate_params function
            self.model_data[self.num_data-1] = inp_data.copy()
            self.estimate_params()
            # Update the state based on estimated model paramters
            self.state = self.model_pars
        else:
            # Update the stored data as well
            self.model_data[1:] = self.model_data[:-1]
            self.model_data[-1] = inp_data


if __name__=="__main__":
    # Order of the motion that we are expecting
    motion_order = 3 # 1 is static, 2 is constant velocity model
    dt = 1.0 # Time steps of dt sec
    const_vel = motion_par(motion_order,dt)
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
    num_points = num_points+10
    prop_state = None
    # Passing this data to the algorithm
    for i in range(inp_data.shape[0]):
        const_vel.process_inp_data(inp_data[i])
        if prop_state is None:
            if const_vel.state is not None:
                prop_state = const_vel.state
        else:
            prop_state = const_vel.propagate(prop_state)
            out_state[i] = prop_state[0]

    inp = plt.plot(inp_data,'b+',label='Input Data',linewidth=2.0)
    out = plt.plot(np.arange(const_vel.min_data_samples,num_points),
                   out_state[const_vel.min_data_samples:],
                   'r', label='State Output', linewidth=2.0)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc='lower left')
    plt.show()

