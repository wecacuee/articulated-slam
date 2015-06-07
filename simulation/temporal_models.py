'''
Implement finite order temporal models of motion
'''
import numpy as np
import math
import matplotlib.pyplot as plt

class temporal_model:
    # Constructor for the class
    def __init__(self,init_state = np.zeros(2)):
        # Define the order of the temporal model
        # Order 1 is the constant velocity model
        self.order = len(init_state)
        # Define the initial state of a finite order temporal model
        self.state = init_state
        # Noise characteristics for the last state derivative, for example for a constant velocity model, we will assume acceleration as noise
        self.noise_sigma = 0.1/len(init_state)
        self.noise_mu = 0 # Assuming zero mean noise

    # Forward propagation model
    def forward_motion(self,dt):
        # Generate a sample for last state by sampling from gaussian with given noise variance
        noise_input = self.noise_mu+self.noise_sigma*np.random.randn()
        # Dummy variable for state prediction
        pred_state = np.zeros(len(self.state))
        for i in range(self.order):
            for j in range(i,self.order):
                pred_state[i] = pred_state[i]+((dt**(j-i))/math.factorial(j-i))*self.state[j]
                #import pdb;pdb.set_trace()
        # Adding noise to the final state
        print i,noise_input
        pred_state[i] = pred_state[i]+noise_input
        self.state = pred_state

if __name__=="__main__":
    init_state = np.array([0,1,1])
    const_vel = temporal_model(init_state)
    dt = 0.1 # Time steps of 0.1 sec
    # Simulate for 10 seconds
    position = np.zeros(int(10.0/dt))
    for i in range(int(10.0/dt)):
        const_vel.forward_motion(dt)
        position[i] = const_vel.state[0]

    plt.plot(position)
    plt.show()



