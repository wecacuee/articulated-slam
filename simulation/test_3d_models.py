""" Test sequence for 3D articulation models
        1. Generate sample point from specific model
        2. Assuming the model is known perform ekf
        3. Observe the predicted motion parameters and ground truth values
        4. Make sure they are close
        5. Add noise and observe outputs """
import numpy as np
import cv2
import landmarkmap
import estimate_mm as mm # To figure out the right motion model
import pdb
import utils_plot as up
import scipy.linalg
import matplotlib.pyplot as plt

def gen_data():
    t = 10*np.pi/180
    R = np.array([[np.cos(t),np.sin(t),0,0],[-np.sin(t),np.cos(t),0,0],[0,0,1,0],[0,0,0,1]])
    P = np.array([0,5,0,1])
    Ptemp = P
    for i in range(25):
       yield R.dot(Ptemp) 
       Ptemp = R.dot(Ptemp) 


def test():
    # Generate the test data
    ldmk_obs =  gen_data()
    
    # Initialize 
    m_thresh = 0.75 # Choose the articulation model with greater than this threhsold's probability
    rob_pos = np.array([0,10,90*np.pi/180]) # Using only x,y,theta

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ld_ids = []
    for obs in ldmk_obs:
        rob_state = rob_pos
        
        motion_class = ldmk_estimater.setdefault(0, mm.Estimate_Mm())

        chosen_am = ldmk_am.setdefault(0,None)

        # For each landmark id, we want to check if the motion model has been estimated
        if ldmk_am[0] is None:
        # Still need to estimate the motion class
            	motion_class.process_inp_data([0,0],rob_state,np.vstack(obs[0:3]),np.array([0,5,0]))
                # Check if the model is estimated
            	if sum(motion_class.prior>m_thresh)>0:
            	   ldmk_am[0] = motion_class.am[np.where(motion_class.prior>m_thresh)[0]]
            	   ld_ids.append(0)
            	   curr_ld_state = ldmk_am[0].current_state()
            	   curr_ld_cov = ldmk_am[0].current_cov()
        else:
		    # Need to convert state values (theta, thetad for revolute) into prediction values of x,y,z
            lk_pred = ldmk_am[0].predict_model(curr_ld_state)		
       	    print lk_pred,obs[0:3]


if __name__ == "__main__":
    test()
