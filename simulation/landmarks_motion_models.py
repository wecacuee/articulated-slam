"""
Given trajectories, estimate the motion models currently being followed by the landmarks
"""
import numpy as np
from scipy.optimize import minimize
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
            self.model_data[1:,:] = self.model_data[:-1,:]
            self.model_data[-1,:] = inp_data
            # We already have a model, maybe we can update the parameters
            self.update_model_pars()
    
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
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.model_par = self.model_par # To Do: Update the parameters location online
        print "Updated Model paramters are", self.model_par
    def predict_model(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Predicting the location of the static landmark by adding gaussian noise
        return np.random.multivariate_normal(self.model_par,noise_cov)


# Landmark motion model that is moving in a circular motion with uniform velocity
class Revolute_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a prismatic model using maximum likelihood
        # Model consists of x_0,y_0,r,theta_0,w_l : center x, center y, radius,start_angle, angular velocity
        # Model is x_k = x_0+r\cos(\theta_0+n*w_l), y_k = y_0+r\sin(\theta_0+n*w_l) where n = 0,...,t are time steps
        print "Fitting the motion model"
        x0 = np.array([0,0,0,0,0])
        res = minimize(self.ml_model,x0,options={'xtol':1e-8,'disp': True})
        self.model_par = res.x
        print "Estimated Model paramters are", self.model_par

    # Defining the maximum likelihood model
    def ml_model(self,x):
        sum_func = 0
        for i in range(self.min_data_samples):
            sum_func = sum_func+(self.model_data[i,0]-((x[2]*np.cos(x[3]+i*x[4]))+x[0]))**2+(self.model_data[i,1]-((x[2]*np.sin(x[3]+i*x[4]))+x[1]))**2
        return sum_func
        
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.model_par = self.model_par # To Do: Update the parameters location online
        print "Updated Model paramters are", self.model_par
    def predict_model(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Predicting the location of the static landmark by adding gaussian noise
        mean_x = self.model_par[2]*np.cos(self.model_par[3]+self.model_par[4]*self.num_data)+self.model_par[0]
        mean_y = self.model_par[2]*np.sin(self.model_par[3]+self.model_par[4]*self.num_data)+self.model_par[1]
        print "Mean Prediction is", mean_x,mean_y
        return np.random.multivariate_normal(np.array([mean_x,mean_y]),noise_cov)


# Landmark motion model that is moving along a line with specified velocity
class Prismatic_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a prismatic model using maximum likelihood
        # Model consists of x_0,y_0,\theta,v_l : Starting x, starting y, slope, velocity along line
        # Model is x_k = n*v_l\cos(\theta)+x_0, y_k = n*v_l\cos(\theta)+y_0 where n = 0,...,t are time steps
        print "Fitting the motion model"
        v_l_guess = np.sqrt((self.model_data[1,0]-self.model_data[0,0])**2+(self.model_data[1,1]-self.model_data[0,1])**2)
        x0 = np.array([self.model_data[0,0],self.model_data[0,1],0,v_l_guess])
        res = minimize(self.ml_model,x0,options={'xtol':1e-8,'disp': True})
        self.model_par = res.x
        print "Estimated Model paramters are", self.model_par

    # Defining the maximum likelihood model
    def ml_model(self,x):
        sum_func = 0
        for i in range(self.min_data_samples):
            sum_func = sum_func+(self.model_data[i,0]-((i*x[3]*np.cos(x[2]))+x[0]))**2+(self.model_data[i,1]-((i*x[3]*np.sin(x[2]))+x[1]))**2
        return sum_func
        
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.model_par = self.model_par # To Do: Update the parameters location online
        print "Updated Model paramters are", self.model_par
    def predict_model(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Predicting the location of the static landmark by adding gaussian noise
        mean_x = self.num_data*self.model_par[3]*np.cos(self.model_par[2])+self.model_par[0]
        mean_y = self.num_data*self.model_par[3]*np.sin(self.model_par[2])+self.model_par[1]
        return np.random.multivariate_normal(np.array([mean_x,mean_y]),noise_cov)


if __name__=="__main__":
    # Lets simulate some data from static sensors
    data = np.array([1,0])
    noise_cov = np.diag([0.05,0.05])
    model1 = Revolute_Landmark(3,noise_cov)
    model1.process_inp_data(data)
    data1 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
    model1.process_inp_data(data1)
    print model1.model_par,model1.model_data
    data2 = np.array([0,1])
    model1.process_inp_data(data2)
    print model1.model_par,model1.model_data,model1.predict_model()

