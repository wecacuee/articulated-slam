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
import pdb

# Abstract class for storing and estimating various motion models
class Articulation_Models:
    # Constructor for each motion model
    '''
    _motion_pars Motion parameters associated with current joint
    _noise_cov : Covariance of noise
    '''
    def __init__(self,_motion_pars,_noise_cov= 1.0):
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

    # Get the motion parameters from state estimates
    def get_motion_pars(self):
        raise NotImplementedError("Subclass must implement its method")




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
    def __init__(self,_temp_order=2,_noise_cov=1.0,_dt=1):
        # For prismatic joint, there is only one motion parameter which
        # is the length along the prismatic axis
        _motion_pars = [mp.Motion_Par(_temp_order,_dt)]
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
        self.config_pars = self.config_pars # To Do: Update the parameters location online
        # Update the motion parameter anyway
        self.motion_pars[0].process_inp_data(self.get_motion_pars(self.model_data[-1,:]))

   
    def predict_motion_pars(self):
        # Predicting just the motion parameters
        pred_motion_pars = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            # Using just the actual prediction of motion parameter and not the
            # velocity, acc or higher derivatives
            pred_motion_pars[i] = self.motion_pars[i].propagate()[0]
        return pred_motion_pars

    def predict_model(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # First predict the motion parameter 
        pred_motion_pars = self.predict_motion_pars()
        # State model is x[k+1] = x_0 + motion_par[0]*t
        
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(self.config_pars,noise_cov)
        
        # Returning just the mean state for now
        return self.config_pars['point']+pred_motion_pars[0]*self.config_pars['normal']
    
    ##
    # @brief Provides linear matrix for EKF filtering
    #       currently only motion parameters are part of the state
    #
    # @return 
    def model_linear_matrix(self):
        return self.motion_pars[0].model_linear_matrix()

    ##
    # @brief Estimate motion paramters from observation data and pass them to motion
    #           parameter objects as well
    #
    # @return 
    def get_motion_pars(self,curr_data):
        # Asserting that we have a model
        assert(self.config_pars is not None), 'No model estimated yet'
        return np.dot(curr_data-self.config_pars['point'],self.config_pars['normal'])  


# Landmark motion model that is moving along a circular path
class Revolute_Landmark(Articulation_Models):
    
    ##
    # @brief Constructor for Revolute Landmark
    #
    # @param _temp_order Temporal order of the revolute motion parameter
    # @param _noise_cov Noise covariance for forward propagation
    # @param _dt Time sampling rate
    #
    # @return 
    def __init__(self,_temp_order=2,_noise_cov=1.0,_dt=1):
        # For revolute joint, there is only one motion parameter which
        # is the angle along the revolute axis
        _motion_pars = [mp.Motion_Par(_temp_order,_dt)]
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
        maxradius, minradius = 1000.0, 0.1

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
        self.config_pars = self.config_pars # To Do: Update the parameters location online
        # Update the motion parameter anyway
        self.motion_pars[0].process_inp_data(self.get_motion_pars(self.model_data[-1,:]))
    
    def predict_motion_pars(self):
        # Predicting just the motion parameters
        pred_motion_pars = [0]*len(self.motion_pars)
        for i in range(len(self.motion_pars)):
            # Using just the actual prediction of motion parameter and not the
            # velocity, acc or higher derivatives
            pred_motion_pars[i] = self.motion_pars[i].propagate()[0]
        return pred_motion_pars    
    
    def predict_model(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        
        # First predict the motion parameter
        pred_motion_pars = self.predict_motion_pars()
        theta = pred_motion_pars[0]
        # State model is x[k+1] = x_0 + [R*cos(theta);R*sin(theta)]
        
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(np.array([mean_x,mean_y]),noise_cov)
        
        # Returning just the mean state for now
        return np.array([self.config_pars['center'][0]+self.config_pars['radius']*np.cos(theta),
            self.config_pars['center'][1]+self.config_pars['radius']*np.sin(theta)])
    
    def model_linear_matrix(self):
        return self.motion_pars[0].model_linear_matrix()

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
        print "Quad is ",self.find_quad(val)," for val ",val
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


# Landmark motion model that is static
class Static_Landmark(Articulation_Models):

    def __init__(self,_noise_cov=1.0):
        _motion_pars = []
        Articulation_Models.__init__(self,_motion_pars,_noise_cov)
    
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a static model using maximum likelihood
        self.config_pars = {'location':self.model_data[-1,:].copy()}
        print "Estimated Model paramters for static model are", self.config_pars
    
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.config_pars = {'location':self.model_data[-1,:].copy()}
    
    def predict_motion_pars(self):
        # There is nothing to do here
        pass

    def predict_model(self):
        # Asserting that we have a model
        assert(self.config_pars is not None),'Do not call this function until we have sufficient data to estimate a model'
        # State model is x[k+1] = x[k] + w_1; y[k+1] = y[k] +w_2 ,
        # where w_1 and w_2 are noise in x and y dir
        
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(self.config_pars,noise_cov)
        
        # Returning just the mean state for now
        return self.config_pars['location']
    
    def model_linear_matrix(self):
        return np.array([[1,0],[0,1]])

    def get_motion_pars(self):
        # Nothing to do here
        pass


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
    
    

