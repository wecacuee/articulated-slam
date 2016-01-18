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

def gen_data(model='rev'):
    # Generate output with initial point appended
    if model=='rev':
        # Model parameters
        r = 1
        w = np.pi/6
        x_0 = 2.0
        y_0 = 2.0
        
        # Case 1: robot @ origin and point is always in view
        # Data assumption is rotation about z-axis since view model hasn't been updated
        #for i in range(30):
        #    yield np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0,1,r*np.cos(0)+x_0,r*np.sin(0)+y_0,1])


        # Case 2: robot is not @ origin but point is always in view
        for i in range(30):
            yield np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0,1,r*np.cos(0)+x_0,r*np.sin(0)+y_0,1])

    elif model=='static':
        x_0 = 2.0
        y_0 = 2.0
        for i in range(30):
            yield np.array([x_0,y_0,1,x_0,y_0,1])

    else:
        x_0 = 2.0
        y_0 = 2.0
        v_y = 0.2
        v_x = 0.1
        for i in range(30):
            yield np.array([x_0+i*v_x,y_0+i*v_y,1,x_0,y_0,1])


def test():
    # Generate the test data
    #ldmk_obs =  gen_data()
    #
    ## Initialize 
    #m_thresh = 0.75 # Choose the articulation model with greater than this threhsold's probability
    #rob_pos = np.array([0,10,90*np.pi/180]) # Using only x,y,theta

    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    #ld_ids = []
    #for obs in ldmk_obs:
    #    rob_state = rob_pos
    #    
    #    motion_class = ldmk_estimater.setdefault(0, mm.Estimate_Mm())

    #    chosen_am = ldmk_am.setdefault(0,None)

    #    # For each landmark id, we want to check if the motion model has been estimated
    #    if ldmk_am[0] is None:
    #    # Still need to estimate the motion class
    #        	motion_class.process_inp_data([0,0],rob_state,np.vstack(obs[0:3]),np.array([0,5,0]))
    #            # Check if the model is estimated
    #        	if sum(motion_class.prior>m_thresh)>0:
    #        	   ldmk_am[0] = motion_class.am[np.where(motion_class.prior>m_thresh)[0]]
    #        	   ld_ids.append(0)
    #        	   curr_ld_state = ldmk_am[0].current_state()
    #        	   curr_ld_cov = ldmk_am[0].current_cov()
    #    else:
    #        import pdb; pdb.set_trace()
	#	    # Need to convert state values (theta, thetad for revolute) into prediction values of x,y,z
    #        lk_pred = ldmk_am[0].predict_model(motion_class.means[0])		
    #   	    print lk_pred,obs[0:3]
    
    data = gen_data('pris')
    # Case 1: robot at origin
    #robot_state = np.array([0,0,0])

    # Case 2: Robot away from origin with some theta
    robot_state = np.array([1,5,np.pi])
    for packet in data:
        
        curr_obs = packet[0:3]
        init_pt = packet[3:]

        motion_class = ldmk_estimater.setdefault(0, mm.Estimate_Mm())

        chosen_am = ldmk_am.setdefault(0,None)

        motion_class.process_inp_data([0,0],robot_state,curr_obs,init_pt)
        if ldmk_am[0] is None:
        # Still estimating model
            if sum(motion_class.prior>0.6)>0:
                model = np.where(motion_class.prior>0.6)[0]
                ldmk_am[0] = motion_class.am[model]
        else:
            #pdb.set_trace() 
            # LK_pred comes out in the robot_frame. We need to convert it back to world frame to match the world coordinate observations
            lk_pred = ldmk_am[0].predict_model(motion_class.means[model])
            
            R_temp = np.array([[np.cos(robot_state[2]), -np.sin(robot_state[2]),0],
                 [np.sin(robot_state[2]), np.cos(robot_state[2]),0],
                 [0,0,1]])
            
            pos_list = np.ndarray.tolist(robot_state[0:2]) 
            pos_list.append(0.0)
                   
            print model,R_temp.T.dot(lk_pred)+np.array(pos_list),curr_obs        

        """ Legacy code (examples)
        # Revolute
        #init_pt,curr_obs = np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0,1])
 
        # Static
        #curr_obs = np.array([x_0,y_0,0])
 
         # Prismatic - starting from x_0,y_0 ,slope of line at w
         #curr_obs = np.array([x_0-i*r*np.cos(w),y_0+i*r*np.sin(w)])
 
        
        #print "Rev: ",motion_class.prior[0],"Pris: ",motion_class.prior[1],"Static: ",motion_class    .prior[2]
        
        #if i>8:
        #    import pdb; pdb.set_trace()
        #    print "Revolute joint observed position ",\
        #    motion_class.am[0].predict_model(motion_class.means[0]), "obs = ",curr_obs"""



if __name__ == "__main__":
    test()
