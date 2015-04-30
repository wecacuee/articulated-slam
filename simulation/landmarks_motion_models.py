"""
Given trajectories, estimate the motion models currently being followed by the landmarks
"""


# Abstract class for storing and estimating various motion models
class Motion_Models:
    def __init__(self,min_pars,noise_var):
        # To store the minimum number of observations needed before pars can be estimated
        self.min_pars = min_pars
        self.noise_var = noise_var # To store the noise parameters of forward motion models
        self.model_par = None # To store the latest updated model parameters
    def fit_model(self,inp_data): # Abstract method
        raise NotImplementedError("Subclass must implement its method")
    def update_model_pars(self,inp_data): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")
    def predict_model(self,inp_data): # Update parameters of the model
        raise NotImplementedError("Subclass must implement its method")

# Landmark motion model that is static
class Static_Landmark(Motion_Models):


# Landmark motion model that is moving in a circular motion with uniform velocity
class Move_Revolute(Motion_Models):

# Landmark motion model that is moving along a line with specified velocity
class Move_Prismatic(Motion_Models):
