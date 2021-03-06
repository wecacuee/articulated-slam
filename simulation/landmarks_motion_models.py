"""
Given trajectories, estimate the motion models currently being followed by the landmarks
"""
import numpy as np
from scipy.optimize import minimize
from scipy.misc import comb
import itertools

# Abstract class for storing and estimating various motion models
class Motion_Models:
    # Constructor for each motion model
    def __init__(self,min_data_samples,noise_cov):
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

    # Derivative of forward motion model evaluated at a particular state
    def model_linear_matrix(self):
        raise NotImplementedError("Subclass must implement its method")

# Landmark motion model that is static
class Static_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a static model using maximum likelihood
        self.model_par = self.model_data[-1,:].copy()
        print "Estimated Model paramters for static model are", self.model_par
    
    def update_model_pars(self):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # Keeping the same parameters for now
        self.model_par = self.model_par # To Do: Update the parameters location online
    
    def predict_model(self,x_k = None):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # State model is x[k+1] = x[k] + w_1; y[k+1] = y[k] +w_2 , where w_1 and w_2 are noise in x and y dir
        if x_k is None:
            # If no previous state is passed, assume its the last data point
            x_k = self.model_data[-1,:]
        mean_x = x_k[0] 
        mean_y = x_k[1]
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(self.model_par,noise_cov)
        
        # Returning just the mean state for now
        return np.array([mean_x,mean_y])
    
    def model_linear_matrix(self):
        return np.array([[1,0],[0,1]])

def revolute_params_analytic(points):
    """
    >>> c = np.ones(2) * 0.5
    >>> r = 1
    >>> thetas = np.array([10, 20, 30]) * np.pi / 180
    >>> points = r * np.vstack((np.cos(thetas), np.sin(thetas))) + c.reshape(-1, 1)
    >>> [round(f, 2) for f in revolute_params_analytic(points)]
    [0.5, 0.5, 1.0, 0.17, 0.17]

    >>> c = np.random.rand(2)
    >>> r = np.random.rand(1)
    >>> omega = np.random.rand(1) * np.pi * 1/3
    >>> theta0 = np.random.rand(1) * np.pi * 1/3
    >>> thetas = theta0 + np.array([0, omega, 2*omega])
    >>> points = r * np.vstack((np.cos(thetas), np.sin(thetas))) + c.reshape(-1, 1)
    >>> out = revolute_params_analytic(points)
    >>> expected = np.hstack((c, r, theta0, omega))
    >>> np.allclose(out, expected)
    True
    """
    # | x_i - c |^2 = r^2, for all i
    # for all i, minimize ||c' c - 2 x_i' c - r^2 ||
    # 
    ## for 2D
    # take any pair of points
    # xm_ij = (x_i + x_j)/2
    # (x_i - xm_ij)' (c - xm_ij) = 0
    # (x_j - xm_ij)' (c - xm_ij) = 0
    # [x_i' - xm_ij'] c = (x_i -xm_ij)' xm_ij
    # [x_j' - xm_ij']   = (x_j -xm_ij)' xm_ij

    #
    ## for 3D
    # take any triplets of points
    # xm_ijk = (xi + xj + xk)/3

    ## for nD
    # take any n points
    d, n = points.shape
    assert n >= d + 1, 'need n > d + 1, n = %d, d = %d' % (n, d)
    nchoosed = comb(n,d)
    A = np.zeros((nchoosed * d, d))
    b = np.zeros(nchoosed * d)
    for i, pt_idx_d_tuple in enumerate(itertools.combinations(range(n), d)):
        mean_point = np.mean(points[:, pt_idx_d_tuple], axis=1)
        for j, pt_idx in enumerate(pt_idx_d_tuple):
            A[d*i + j, :] = points[:, pt_idx] - mean_point
            b[d*i + j] = (points[:, pt_idx] - mean_point).dot(mean_point)


    c = np.linalg.lstsq(A,b)[0]
    cpts = points - c.reshape(-1,1)
    r = np.mean(np.sqrt(np.sum(cpts**2, axis=0)))
    thetas = np.arctan2(cpts[1,:], cpts[0,:]) # only valid for 2D
    omega = np.mean(np.diff(thetas))
    retvals = list(c)
    retvals.extend([r, thetas[0], omega])
    return retvals



# Landmark motion model that is moving in a circular motion with uniform velocity
class Revolute_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a prismatic model using maximum likelihood
        # Model consists of x_0,y_0,r,theta_0,w_l : center x, center y, radius,start_angle, angular velocity
        # Model is x_k = x_0+r\cos(\theta_0+n*w_l), y_k = y_0+r\sin(\theta_0+n*w_l) where n = 0,...,t are time steps
        # x0 = np.array([0,0,0,0,0])
        points = self.model_data[:self.min_data_samples, :].T
        x0 = revolute_params_analytic(points)
        if np.any(np.isinf(x0)) or np.any(np.isnan(x0)):
            x0 = np.array([0, 0, 0, 0, 0])

        # x[2] is radius -- maximum radius 100, min radius 1
        maxradius, minradius = 1000.0, 0.1
        # x[4] is angular velocity -- minimum 0.1 rad/delta T
        minangvel = 0.01
        cons = ({'type':'ineq','fun': lambda x:maxradius-x[2]},
                {'type':'ineq','fun': lambda x:x[2]-minradius},
                {'type':'ineq','fun': lambda x: np.abs(x[4])-minangvel})
        res = minimize(self.ml_model,x0,method='SLSQP',constraints = cons)
        self.model_par = res.x
        print "Estimated Model paramters for revolute are", self.model_par
        #print "Input data was",self.model_data

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
        #print "Updated Model paramters are", self.model_par
    def predict_model(self,x_k = None):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        if x_k is None:
            # If no previous state is passed, assume its the last data point
            x_k = self.model_data[-1,:]
        
        # State model is x[k+1] = x[k]*cos(w)-y[k]*sin(w)-x_0*cos(w)+y_0*sin(w)+x_0 + w_1;
        # y[k+1] = x[k]*sin(w)+y[k]*cos(w)-x_0*sin(w)-y_0*cos(w)+y_0 + w_2 ,
        # where w_1 and w_2 are noise in x and y dir
        w = self.model_par[4]
        x_0 = self.model_par[0]
        y_0 = self.model_par[1]
        mean_x = x_k[0]*np.cos(w)-x_k[1]*np.sin(w)-x_0*np.cos(w)+y_0*np.sin(w)+x_0
        mean_y = x_k[0]*np.sin(w)+x_k[1]*np.cos(w)-x_0*np.sin(w)-y_0*np.cos(w)+y_0
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(np.array([mean_x,mean_y]),noise_cov)
        '''
        mean_x = self.model_par[2]*np.cos(self.model_par[3]+self.model_par[4]*self.num_data)+self.model_par[0]
        mean_y = self.model_par[2]*np.sin(self.model_par[3]+self.model_par[4]*self.num_data)+self.model_par[1]
        '''
        # Returning just the mean state for now
        return np.array([mean_x,mean_y])
    
    def model_linear_matrix(self):
        w = self.model_par[4]
        return np.array([[np.cos(w),-np.sin(w)],[np.sin(w),np.cos(w)]])

    def get_revolute_par(self):
        # Model consists of x_0,y_0,r,theta_0,w_l : center x, center y, radius,start_angle, angular velocity
        x = self.model_par
        center = x[0:2]
        radius = x[2]
        theta_0 = x[3]
        omega = x[4]
        return (center, radius, theta_0, omega)


# Landmark motion model that is moving along a line with specified velocity
class Prismatic_Landmark(Motion_Models):
    # Model paramters for a static landmark is just the location of the landmark
    def fit_model(self):
        # asserting that we have atleast the minimum required observation for estimation
        assert(self.num_data>=self.min_data_samples),"Can not call this function until we have sufficient data to estimate the model"
        # Fitting a prismatic model using maximum likelihood
        # Model consists of x_0,y_0,\theta,v_l : Starting x, starting y, slope, velocity along line
        # Model is x_k = n*v_l\cos(\theta)+x_0, y_k = n*v_l\sin(\theta)+y_0 where n = 0,...,t are time steps
        print "Fitting the motion model"
        v_l_guess = np.sqrt((self.model_data[1,0]-self.model_data[0,0])**2+(self.model_data[1,1]-self.model_data[0,1])**2)
        x0 = np.array([self.model_data[0,0],self.model_data[0,1],0,v_l_guess])
        
        # Vikas: Please verify what should we use as minimum
        # We need a lower threshold on linear velocity because otherwise a static landmark is a prismatic with zero velocity in any direction
        # Adding a minimum value of absolute of velocity
        minabsvel = 1
        cons = ({'type':'ineq','fun':lambda x: np.abs(x[3])-minabsvel})

        res = minimize(self.ml_model,x0,method = 'SLSQP',constraints=cons)


        self.model_par = res.x
        print "Estimated Model paramters for prismatic are", self.model_par

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
        #print "Updated Model paramters are", self.model_par
    def predict_model(self,x_k = None):
        # Asserting that we have a model
        assert(self.model_par is not None),'Do not call this function until we have sufficient data to estimate a model'
        # State model is x[k+1] = x[k] + v_l*cos(theta)+ w_1;
        # y[k+1] = y[k] + v_l*sin(theta)+ w_2 , where w_1 and w_2 are noise in x and y dir
        if x_k is None:
            # If no previous state is passed, assume its the last data point
            x_k = self.model_data[-1,:]
        v_l = self.model_par[3]
        theta = self.model_par[2]
        mean_x = x_k[0] + v_l*np.cos(theta)
        mean_y = x_k[1] + v_l*np.sin(theta)
        # Forward Sample -- Includes noise
        # np.random.multivariate_normal(self.model_par,noise_cov)
        
        # Returning just the mean state for now
        return np.array([mean_x,mean_y])
    
    def model_linear_matrix(self):
        return np.array([[1,0],[0,1]])

    def get_prismatic_par(self):
        # Model consists of x_0,y_0,\theta,v_l : Starting x, starting y, slope, velocity along line
        x = self.model_par
        x0 = x[0:2]
        delx = x[3] * np.array([np.cos(x[2]), np.sin(x[2])])
        return x0, delx


if __name__=="__main__":
    # Lets simulate some data from static sensors
    data = np.array([3,2])
    data1 = np.array([2+np.cos(np.pi/6),2+np.sin(np.pi/6)])
    data2 = np.array([2+np.cos(np.pi/3),2+np.sin(np.pi/3)])
    data3 = np.array([2+np.cos(np.pi*0.4),2+np.sin(np.pi*0.4)])
    noise_cov = np.diag([0.01,0.01])
    
    model1 = Revolute_Landmark(3,noise_cov)
    model1.process_inp_data(data)
    model1.process_inp_data(data1)
    model1.process_inp_data(data3)
    print model1.model_par,model1.predict_model()
    print "New prediction ", model1.predict_model(np.array([3,2]))

    '''
    model2 = Prismatic_Landmark(2,noise_cov)
    model2.process_inp_data(data)
    model2.process_inp_data(data1)
    model2.process_inp_data(data2)
    print model2.model_par,model2.model_data,model2.predict_model(),model2.model_linear_matrix()
    '''
    

