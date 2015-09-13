"""
All articulated motion in the world can be represented as 
X(t) = f(C,m(t))
where X(t) is the observed motion of an object
C is the configuration space (doors open about an axis)
m(t) is the motion parameter (Ex: length of prismatic joint, 
angle of door)

This class models the configuration space of articulated joints.
In particular, it implements various 2D and 3D commonly possible 
articulated joints such as prismatic, static, revolute etc.

This code is based upon the code in landmarks_motion_models.py
The only difference is that it explicitly models only the spatial
part and imports motion_par class for temporal parameters.
"""

import numpy as np
from scipy import optimize 
from scipy.misc import comb
import itertools
import motion_par as mp # Motion parameters
import scipy.linalg
import pdb

# Abstract class for storing and estimating various motion models
class Articulation_Models:
    # Constructor for each motion model
    '''
    _motion_pars Motion parameters associated with current joint
    _noise_cov : Covariance of noise
    '''
    def __init__(self,_motion_pars,_noise_cov= 1.0,_min_speed=0.1):
        # Defining the motion parameters for the current articulation model
        self.motion_pars = _motion_pars
        self.noise_cov = _noise_cov # To store the noise covariance of motion models
        # To store the minimum number of observations needed before all the 
        # motion parameters can be estimated, assumes that _motion_pars is a list
        if len(self.motion_pars)>0:
            self.min_data_samples = max(1,max(mpar.min_data_samples \
                    for mpar in self.motion_pars)) 
        else:
            # No motion parameters are involved
            self.min_data_samples = 1

        self.config_pars = None # To store the configuration parameters of articulated joint
        self.num_data = 0 # Number of observation samples for current landmark
        self.model_data = None
    
    # Main function to take in streaming data 
    def process_inp_data(self,inp_data):
        if self.model_data is None:
            # Initializing the model_data dimension from the first observation
            self.model_data = np.zeros((self.min_data_samples,inp_data.shape[0]))
        # Main function to recieve input data and call appropriate functions
        self.num_data = self.num_data+1
        if self.num_data<self.min_data_samples:
            # Add data to model data
            self.model_data[self.num_data-1,:] = inp_data.copy()
        elif self.num_data==self.min_data_samples:
            # Add data and call fit_model function
            self.model_data[self.num_data-1,:] = inp_data.copy()
            self.fit_model()
        else:
            # Update the stored data as well
            # Using the first in first out principle to store the latest data only
            self.model_data[:-1,:] = self.model_data[1:,:]
            self.model_data[-1,:] = inp_data
            # We already have a model, maybe we can update the parameters
            self.update_model_pars()
    
    # To get the current motion parameter states of the each articulated model 
    def current_state(self,assemble=1):
        current_state = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            current_state[i] = self.motion_pars[i].state
        if assemble is not None:
            current_state = np.ravel(np.asarray(current_state))
        return current_state

    # To get the current motion parameter covariance of the each articulated model 
    def current_cov(self,assemble=1):
        # Get current state
        current_state = self.current_state()
        current_cov = self.noise_cov*np.diag(np.ones(current_state.shape[0]))
        return current_cov
    
    # Fit the current spatial motion model to the input data
    def fit_model(self): # Abstract method
        raise NotImplementedError("Subclass must implement its method")

    # To update the model parameters
    def update_model_pars(self): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")

    # Forward motion model incorporating both configuration and 
    # motion prediction
    def predict_model(self): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")

    # Motion prediction step for just the motion parameters
    def predict_motion_pars(self): # Predict just the motion parameters
        raise NotImplementedError("Subclass must implement its method")

    # Derivative of forward motion model evaluated at a particular state
    def model_linear_matrix(self):
        raise NotImplementedError("Subclass must implement its method")

    # Define observation jacobian
    def observation_jac(self):
        raise NotImplementedError("Subclass must implement its method")

    # Get the motion parameters from state estimates
    def get_motion_pars(self):
        raise NotImplementedError("Subclass must implement its method")

    # To propagate the covariance of motion parameter from current frame to next frame
    def prop_motion_par_cov(self,inp_cov):
        start_ind = 0
        for motion_par in self.motion_pars:
            # Getting the linear matrix from each motion par matrix
            lin_mat = motion_par.model_linear_matrix()
            end_ind = start_ind+lin_mat.shape[0]
            # Doing the propagation
            prop_cov = np.dot(lin_mat,np.dot(inp_cov[start_ind:end_ind,start_ind:end_ind],lin_mat.T))+\
                    motion_par.model_qmat()
            # Actually updating the inp_cov
            inp_cov[start_ind:end_ind,start_ind:end_ind] = prop_cov
            # Updating for indexing
            start_ind = end_ind
        return inp_cov


# Landmark motion model that is moving along a line 
class Prismatic_Landmark(Articulation_Models):

    ##
    # @brief Constructor for Prismatic Landmark
    #
    # @param _temp_order Temporal order of the prismatic motion parameter
    # @param _noise_cov Noise covariance for forward propagation
    # @param _dt Time sampling rate
    #
    # @return 
    def __init__(self,_temp_order=2,_noise_cov=1.0,_dt=1,_min_speed=0.1):
        # For prismatic joint, there is only one motion parameter which
        # is the length along the prismatic axis
        _motion_pars = [mp.Motion_Par(_temp_order,_dt,_min_speed)]
        Articulation_Models.__init__(self,_motion_pars,_noise_cov)

    ##
    # @brief Fit the spatial model first, estimate the corresponding temporal part and
    #        then fit the temporal part as well
    #
    # @return 
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Need more data to estimate model"
        # Fitting a line using least square estimates
        # Model consists of  x_0 (point on line), t (unit vector along line)
        
        # The point on the line with least square criteria is simply
        # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        x0 = self.model_data.mean(axis=0)
        # Getting the unit vector along line
        uu,dd,vv = np.linalg.svd(self.model_data-x0)
        # I have no idea but we have to do this to get this to work
        if vv[0][0]<0: # Theoretically it doesn't matter but the code cares
            vv[0] = -1*vv[0]
        self.config_pars = {'point':x0,'normal':vv[0]}
        print "Estimated configuration paramters for prismatic are", self.config_pars
        # Now estimate the motion parameters as well
        for curr_data in self.model_data:
            self.motion_pars[0].process_inp_data(self.get_motion_pars(curr_data))
        # Done estimating the motion parameters as well

        
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        #self.config_pars = self.config_pars # To Do: Update the parameters location online
        # Update the motion parameter anyway
        self.motion_pars[0].process_inp_data(self.get_motion_pars(self.model_data[-1,:]))

   
    ##
    # @brief To predict the motion parameters from current state to the next state
    #
    # @param inp_state Optional input state for state prediction
    # @param assemble EKF like algorithms prefer an array instead of list of arrays
    #
    # @return Predicted Motion parameters which is either an assembled array or list of arrays
    def predict_motion_pars(self,inp_state=None,assemble=1):
        # Predicting just the motion parameters
        pred_motion_pars = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            # Using just the actual prediction of motion parameter and not the
            # velocity, acc or higher derivatives
            pred_motion_pars[i] = self.motion_pars[i].propagate(inp_state)
        # Assembling if the assebly is required
        if assemble is not None:
            pred_motion_pars = np.ravel(np.asarray(pred_motion_pars))

        return pred_motion_pars


    # To predict the actual location of landmark
    def predict_model(self,inp_state=None):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        if inp_state is None:
            # First predict the motion parameter
            # Without assembling
            pred_motion_pars = self.predict_motion_pars(assemble=None)
            displacement = pred_motion_pars[0]
            # State model is x[k+1] = x_0 + motion_par[0]*t
            # Forward Sample -- Includes noise
            # np.random.multivariate_normal(self.config_pars,noise_cov)
        else:
            displacement = inp_state[0]
        
        return self.config_pars['point']+displacement*self.config_pars['normal']
    
    ##
    # @brief Provides linear matrix for EKF filtering
    #       currently only motion parameters are part of the state
    #
    # @return 
    def model_linear_matrix(self,assemble=1):
        mmat = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            mmat[i] = self.motion_pars[i].model_linear_matrix()
        if assemble is not None:
            mmat = scipy.linalg.block_diag(*mmat)

        return mmat

    # The only observation we make are (x,y) position and the variable
    # with respect to which we take derivative is the position along the prismatic joint
    def observation_jac(self,inp_state):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        mat = np.zeros((inp_state.shape[0],self.motion_pars[0].order))
        for i in range(mat.shape[0]):
            mat[i,0] = self.config_pars['normal'][i]
        return mat
    
    ##
    # @brief Estimate motion paramters from observation data and pass them to motion
    #           parameter objects as well
    #
    # @return 
    def get_motion_pars(self,curr_data):
        # Asserting that we have a model
        assert(self.config_pars is not None), 'No model estimated yet'
        return np.dot(curr_data-self.config_pars['point'],self.config_pars['normal'])
    
    # Only used for plotting axis and things
    def get_prismatic_par(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),"No model estimated yet"
        return self.config_pars['point'],self.config_pars['normal']*self.motion_pars[0].model_pars[-1]


# Landmark motion model that is moving along a circular path
class Revolute_Landmark(Articulation_Models):
    
    ##
    # @brief Constructor for Revolute Landmark
    #
    # @param _temp_order Temporal order of the revolute motion parameter
    # @param _noise_cov Noise covariance for forward propagation
    # @param _dt Time sampling rate
    # @param _min_speed Minimum speed in terms of the angle (0.02 Radian is 1.14 degrees)
    #
    # @return 
    def __init__(self,_temp_order=2,_noise_cov=1.0,_dt=1,_min_speed=0.02):
        # For revolute joint, there is only one motion parameter which
        # is the angle along the revolute axis
        _motion_pars = [mp.Motion_Par(_temp_order,_dt,_min_speed)]
        self.addition_factor = 0
        Articulation_Models.__init__(self,_motion_pars,_noise_cov)

    # Defining functions for estimating the spatial model
    def calc_R(self,xc,yc):
        # Calculate the distance of each 2D points from the center (xc,yc)
        return np.sqrt((self.model_data[:,0]-xc)**2+(self.model_data[:,1]-yc)**2)

    def f_2(self,c):
        # Calculate the algebraic distance between the data points and mean circle
        Ri = self.calc_R(*c)
        return Ri-Ri.mean()

    ##
    # @brief First fit the circle to the revolute data and then estimate the temporal
    #           part and fit the temporal part as well
    #
    # @return 
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Need more data to estimate the model"
        # First fitting the spatial model
        
        # Algorithm source http://wiki.scipy.org/Cookbook/Least_Squares_Circle
        # First computing center of the circle
        circle_center, ier = optimize.leastsq(self.f_2,np.array([np.mean(self.model_data[:,0]),
            np.mean(self.model_data[:,1])]))
        # Calculating radius of the circle
        Ri_2 = self.calc_R(*circle_center)
        R_2 = Ri_2.mean()
        
        # We need to separate the real revolute joint from the static and prismatic joint
        maxradius, minradius = 1000.0, 0.8

        # To separate it from static joint, fix minimum revolute joint radius
        if R_2<minradius:
            R_2 = minradius
        # To separate it from prismatic joint, fix maximum revolute joint radius
        if R_2>maxradius:
            R_2 = maxradius

        # Storing the model parameters: Circle Center and Radius
        self.config_pars = {'center':np.array([circle_center[0],circle_center[1]]),
            'radius':R_2}
        print "Estimated Model paramters for revolute are", self.config_pars

        # Now estimating the motion parameter as well
        for curr_data in self.model_data:
            self.motion_pars[0].process_inp_data(self.get_motion_pars(curr_data))
        # Done estimating the motion parameters as well

        
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        #self.config_pars = self.config_pars # To Do: Update the parameters location online
        # Update the motion parameter anyway
        self.motion_pars[0].process_inp_data(self.get_motion_pars(self.model_data[-1,:]))
    
    def predict_motion_pars(self,inp_state=None,assemble=1):
        # Predicting just the motion parameters
        pred_motion_pars = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            # Using just the actual prediction of motion parameter and not the
            # velocity, acc or higher derivatives
            pred_motion_pars[i] = self.motion_pars[i].propagate(inp_state)

        # Assembling if the assebly is required
        if assemble is not None:
            pred_motion_pars = np.ravel(np.asarray(pred_motion_pars))
        
        return pred_motion_pars    
    
    def predict_model(self,inp_state=None):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        if inp_state is None:        
            # First predict the motion parameter
            pred_motion_pars = self.predict_motion_pars(assemble=None)
            theta = pred_motion_pars[0]
            # State model is x[k+1] = x_0 + [R*cos(theta);R*sin(theta)]
        
            # Forward Sample -- Includes noise
            # np.random.multivariate_normal(np.array([mean_x,mean_y]),noise_cov)
        else:
            theta = inp_state[0]
        
        # Returning just the mean state for now
        return np.array([self.config_pars['center'][0]+self.config_pars['radius']*np.cos(theta),
            self.config_pars['center'][1]+self.config_pars['radius']*np.sin(theta)])
    
    def model_linear_matrix(self,assemble=1):
        mmat = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            mmat[i] = self.motion_pars[i].model_linear_matrix()
        if assemble is not None:
            mmat = scipy.linalg.block_diag(*mmat)

        return mmat

    def observation_jac(self,inp_state):
        # Assuming that first input state is the angle \theta
        # Asserting that we have a model
        assert(self.config_pars is not None),'No model estimated yet'
        mat = np.zeros((2,self.motion_pars[0].order))
        mat[0,0] = -self.config_pars['radius']*np.sin(inp_state[0])
        mat[1,0] = self.config_pars['radius']*np.cos(inp_state[0])
        return mat


    def find_quad(self,angle):
        angle = (angle)%(2*np.pi)
        if angle<=np.pi/2:
            quad = 1
        elif (np.pi/2<angle) and (angle<=np.pi):
            quad = 2
        elif (np.pi<angle) and (angle<=3*(np.pi/2)):
            quad = 3
        else:
            quad = 4
        return quad

    ##
    # @brief Estimate the motion parameter (angle) given the point on the circle
    #
    # @param curr_data Current observation
    #
    # @return 
    def get_motion_pars(self,curr_data):
        # Asserting that we have a model
        assert(self.config_pars is not None),'No model estimated yet'
        val = np.arctan2(curr_data[1]-self.config_pars['center'][0],
                curr_data[0]-self.config_pars['center'][1])
        if val<0:
            val = val+2*np.pi
        # Need to take special care because angle wraps back at pi
        if self.motion_pars[0].num_data>0:
            if self.motion_pars[0].num_data<self.motion_pars[0].min_data_samples:
                last_val = self.motion_pars[0].model_data[self.motion_pars[0].num_data-1]
            else:
                last_val = self.motion_pars[0].model_data[-1]
            if abs(last_val-val)>3:
                val = val+self.addition_factor
                if abs(last_val-val)>3:
                    if ((self.find_quad(val)==4 and self.find_quad(last_val)<3)) or \
                            ((self.find_quad(val)==3) and (self.find_quad(last_val)==1)):
                        self.addition_factor = self.addition_factor-2*np.pi
                        val = val-2*np.pi
                    else:
                        self.addition_factor = self.addition_factor+2*np.pi
                        val = val+2*np.pi
                    
        return val

    def get_revolute_par(self):
        assert(self.config_pars is not None),'No model estimated yet'
        center = self.config_pars['center']
        radius = self.config_pars['radius']
        theta_0 = self.motion_pars[0].model_data[0] # To get the first angle
        omega = self.motion_pars[0].model_pars[-1] # To modify later on
        return (center,radius,theta_0,omega)


# Landmark motion model that is static
class Static_Landmark(Articulation_Models):

    def __init__(self,_noise_cov=1.0,_dt=1):
        # Minimum order motion parameter
        _motion_pars = [mp.Motion_Par(1,_dt)]
        Articulation_Models.__init__(self,_motion_pars,_noise_cov)
    
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a static model using maximum likelihood
        self.config_pars = {'location':self.model_data[-1,:].copy()}
        print "Estimated Model paramters for static model are", self.config_pars
        # Assuming the model is x_{t+1} = x_{t}+\tau,y_{t+1} = y_{t}+\tau where \tau is zero
        for curr_data in self.model_data:
            self.motion_pars[0].process_inp_data(self.get_motion_pars(curr_data))
    
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        # self.config_pars = {'location':self.model_data[-1,:].copy()}
        # Update the motion parameter anyway
        self.motion_pars[0].process_inp_data(self.get_motion_pars(self.model_data[-1,:]))
    
    def predict_motion_pars(self,inp_state=None,assemble=1):
       # Predicting just the motion parameters
        pred_motion_pars = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            # Using just the actual prediction of motion parameter and not the
            # velocity, acc or higher derivatives
            pred_motion_pars[i] = self.motion_pars[i].propagate(inp_state)

        # Assembling if the assebly is required
        if assemble is not None:
            pred_motion_pars = np.ravel(np.asarray(pred_motion_pars))
        return pred_motion_pars 

    def predict_model(self,inp_state=None):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # State model is x[k+1] = x[k] + w_1; y[k+1] = y[k] +w_2 ,
        # where w_1 and w_2 are noise in x and y dir
        
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(self.config_pars,noise_cov)
        
        # Returning just the mean state for now
        return self.config_pars['location']
    
    def model_linear_matrix(self,assemble=1):
        mmat = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            mmat[i] = self.motion_pars[i].model_linear_matrix()
        if assemble is not None:
            mmat = scipy.linalg.block_diag(*mmat)

        return mmat

    
    def observation_jac(self,inp_state):
        mat = np.zeros((2,self.motion_pars[0].order))
        mat[0,0] = 1 
        mat[1,0] = 1
        return mat
    
    def get_motion_pars(self,curr_data):
        # Asserting that we have a model
        assert(self.config_pars is not None),'No model estimated yet'
        # Its a static model, the deviation from location is supposed to be zero ideally
        return 0


if __name__=="__main__":
    # Lets simulate some data from static sensors
    data = np.array([3,2])
    data1 = np.array([2+np.cos(np.pi/6),2+np.sin(np.pi/6)])
    data2 = np.array([2+np.cos(np.pi/3),2+np.sin(np.pi/3)])
    data3 = np.array([2+np.cos(np.pi/2),2+np.sin(np.pi/2)])
    noise_cov = np.diag([0.01,0.01])
    
    model1 = Static_Landmark()
    for i in range(8):
        model1.process_inp_data(np.array([2*np.cos((np.pi*i)/4),2*np.sin((np.pi*i)/4)]))
    pdb.set_trace()
    '''
    model1.process_inp_data(data)
    model1.process_inp_data(data1)
    model1.process_inp_data(data2)
    model1.process_inp_data(data3)
    print model1.model_par,model1.predict_model()
    print "New prediction ", model1.predict_model(np.array([3,2]))

    
    model2 = Prismatic_Landmark(2,noise_cov)
    pdb.set_trace()
    model2.process_inp_data(data)
    model2.process_inp_data(data1)
    model2.process_inp_data(data2)
    print model2.model_par,model2.model_data,model2.predict_model(),model2.model_linear_matrix()
    '''
    
    

