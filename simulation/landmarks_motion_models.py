"""
Given trajectories, estimate the motion models currently being followed by the landmarks
"""
import numpy as np
import pdb

# Abstract class for storing and estimating various motion models
class Motion_Models:
    # Constructor for each motion model
    def __init__(self,min_data_samples,noise_var):
        # To store the minimum number of observations needed before pars can be estimated
        self.min_data_samples = min_data_samples 
        self.noise_cov = noise_cov # To store the noise covariance of motion models
        self.model_par = None # To store the latest updated model parameters
        self.num_data = 0 # Number of data samples for current landmark
        self.model_data = np.zeros((min_data_samples,2))
    
    # Main function to take in streaming data 
    def process_inp_data(self,inp_data): 
        print "Before Model paramters are", self.model_par
        # Main function to recieve input data and call appropriate functions
        self.num_data = self.num_data+1
        if self.num_data<self.min_data_samples:
            # Add data to model data
            self.model_data[self.num_data-1,:] = inp_data
            print "Just Adding data"
        elif self.num_data==self.min_data_samples:
            # Add data and call fit_model function
            self.model_data[self.num_data-1,:] = inp_data
            print "Calling fitting model"
            self.fit_model()
        else:
            # Update the stored data as well
            self.model_data[1:,:] = self.model_data[:-1,:]
            self.model_data[-1,:] = inp_data
            print "Calling update model"
            # We already have a model, maybe we can update the parameters
            self.update_model_pars()
        print "After Model paramters are", self.model_par
    
    # Fit the current motion model to the input data
    def fit_model(self): # Abstract method
        raise NotImplementedError("Subclass must implement its method")

    # To update the model parameters
    def update_model_pars(self): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")

    # Forward motion model
    def predict_model(self): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")

# Landmark motion model that is static
class Static_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a static model using maximum likelihood
        print "Fitting the motion model"
        self.model_par = self.model_data[-1,:].copy()
        print "Estimated Model paramters are", self.model_par
    def update_model_pars(self):
        print "Input for Updated Model paramters are", self.model_par
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this functin until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.model_par = self.model_par # To Do: Update the parameters location online
        print "Updated Model paramters are", self.model_par
    def predict_model(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this functin until we have sufficient data to estimate a model'
        # Predicting the location of the static landmark by adding gaussian noise
        return np.random.multivariate_normal(self.model_par,noise_cov)


'''
# Landmark motion model that is moving in a circular motion with uniform velocity
class Move_Revolute(Motion_Models):
    def fit_model()

# Landmark motion model that is moving along a line with specified velocity
class Move_Prismatic(Motion_Models):
'''

if __name__=="__main__":
    # Lets simulate some data from static sensors
    data = np.array([1,2])
    noise_cov = np.diag([0.1,0.1])
    model1 = Static_Landmark(1,noise_cov)
    model1.process_inp_data(data)
    print model1.model_par,model1.predict_model(),model1.model_data
    data1 = np.array([1.2,3.2])
    model1.process_inp_data(data1)
    print model1.model_par
    data2 = np.array([1.1,2.5])
    model1.process_inp_data(data2)
    print model1.model_par

