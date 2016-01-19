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
import itertools
from mpl_toolkits.mplot3d import axes3d

def gen_init(model='rev'):
    if model=='rev':
        r =  1
        w = np.pi/6
        x_0 = 0.0
        y_0 = 2.0
        return np.array([r*np.cos(0)+x_0,r*np.sin(0)+y_0,1])
    elif model=='static':
        return np.array([2.0,2.0,0.0])
    else:
        return np.array([0.0,2.0,0.0])


def gen_simple_data(model='rev'):
    # Generate output with initial point appended
    if model=='rev':
        # Model parameters
        r = 1
        w = np.pi/6
        x_0 = 0.0
        y_0 = 2.0
        
        # Case 1: robot @ origin and point is always in view
        # Data assumption is rotation about z-axis since view model hasn't been updated
        #for i in range(30):
        #    yield np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0,1,r*np.cos(0)+x_0,r*np.sin(0)+y_0,1])


        # Case 2: robot is not @ origin but point is always in view
        for i in range(30):
            yield np.array([r*np.cos(-i*w)+x_0,r*np.sin(-i*w)+y_0,1])

    elif model=='static':
        x_0 = 2.0
        y_0 = 2.0
        z_0 = 0.0
        for i in range(30):
            yield np.array([x_0,y_0,z_0])

    else:               # For point moving out/in  view
        x_0 = 2.0       #0/3
        y_0 = 2.0       #2/2
        v_y = 0.1       #0/0
        v_x = 0.1       #0.1/-0.1
        for i in range(30):
            yield np.array([x_0+i*v_x,y_0+i*v_y,0])

def in_view(rob_state,ldmk_pos):
    #pt = np.array([2.0,2.0,0.0])
    pt = ldmk_pos
    #apex = np.array([1.0,5.0,0.0])
    apex = np.array([rob_state[0],rob_state[1],0.0])
    max_dist = 20
    phi_view = 90*np.pi/180 # Assumption of robot looking in plane (flat) 
    theta_view = rob_state[2] # robot_state[2] 
    aperture = 45*np.pi/180
    basement = np.array([max_dist*np.cos(theta_view)*np.sin(phi_view),
                  max_dist*np.sin(theta_view)*np.sin(phi_view),
                 max_dist*np.cos(phi_view)])
                  
    basement = basement + np.array(apex)
  
      
    ap2vec = apex - pt
    axisvec = apex - basement
     
    X = np.around(ap2vec.dot(axisvec)/np.linalg.norm(axisvec)/np.linalg.norm(ap2vec),3) >= np.around(np.cos(aperture),3)
  
    Y = np.around(ap2vec.dot(axisvec)/np.linalg.norm(axisvec),3) <= np.around(np.linalg.norm(axisvec))
    return X and Y


def gen_id():
    for i in range(30):
        yield np.array([0,1])    
                    
def gen_cmplx_data(model1='rev',model2='rev'):
    data = gen_simple_data(model1)
    
    data2 = gen_simple_data(model2) 
    
    for pt1,pt2 in zip(data,data2):
        yield np.vstack((pt1,pt2))

def gen_init_coll(pt1,pt2):
    for i in range(30):
        yield np.vstack((pt1,pt2))

def plot_show():
    fig = plt.figure()
    return fig,plt

def visualize_new(colors,pt,plot_data):
    col = ['k','r','y','b']
    for i in range(pt.shape[0]):
        X,Y,Z = [pt[i][0],pt[i][1],pt[i][2]]
        plot_data.append((X,Y,Z,col[colors[i]],'o'))
    return plot_data

def disp(plot_data):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    for X,Y,Z,ccol,mrk in plot_data:
        ax.scatter(X,Y,Z,c=ccol,marker=mrk)    

    plt.show()


def test():
    # For visualization
    joint_color = [np.array(l) for l in (
                                        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    plot_data =[(0,0,0,'k','o')] 
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
    model = [0,0]
    error = [0.0,0.0]
    error_count = [0,0]

    # Need to modify data generation
    data = gen_cmplx_data('pris','rev')
    data_init_pt = gen_init('pris')
    data2_init_pt = gen_init('rev')   
    
    
    ids = gen_id()
    init_pt = gen_init_coll(data_init_pt,data2_init_pt) 

    # Case 1: robot at origin
    robot_state = np.array([0,0,np.pi*90/180])

    # Case 2: Robot away from origin with some theta
    #robot_state = np.array([1,5,np.pi*90/180])

    # Noise
    mu = 0.0
    sigma = 0.1
    noise = np.random.normal(mu,sigma,3)
    print "Noise is :", noise 
    

    for fidx,data,init_point in zip(ids,data,init_pt):
        

        colors = []
        m_thresh = 0.6
        for fid,dat,init_pt in itertools.izip(fidx,data,init_point):
            curr_obs = dat
            #curr_obs = curr_obs + noise 
        
            if in_view(robot_state,curr_obs):
                motion_class = ldmk_estimater.setdefault(fid, mm.Estimate_Mm())

                chosen_am = ldmk_am.setdefault(fid,None)

                motion_class.process_inp_data([0,0],robot_state,curr_obs,init_pt)
                if ldmk_am[fid] is None:
                # Still estimating model
                    if sum(motion_class.prior>m_thresh)>0:
                        model[fid] = np.where(motion_class.prior>m_thresh)[0]
                        ldmk_am[fid] = motion_class.am[model[fid]]
                else:
                    # LK_pred comes out in the robot_frame. We need to convert it back to world frame to match the world coordinate observations
                    lk_pred = ldmk_am[fid].predict_model(motion_class.means[model[fid]])
                    
                    R_temp = np.array([[np.cos(-robot_state[2]), -np.sin(-robot_state[2]),0],
                         [np.sin(-robot_state[2]), np.cos(-robot_state[2]),0],
                         [0,0,1]])
                    
                    pos_list = np.ndarray.tolist(robot_state[0:2]) 
                    pos_list.append(0.0)
                    res = R_temp.T.dot(lk_pred)+np.array(pos_list)
                    error[fid] = error[fid] + np.linalg.norm(res - curr_obs)
                    error_count[fid] = error_count[fid] + 1
            else:
                print "Not in view"
        
            ## Visualization
            track = 1
            for prob in ldmk_estimater[fid].prior[:3]:
                if prob > m_thresh:
                    colors.append(track)
                    break
                track = track + 1
            if track == 4:
                colors.append(0)
            #color = np.int64((p1*rev_color
            #                   + p2*pris_color
            #                   + p3*stat_color))
            #color = color - np.min(color)
            #colors.append(color)

        # Visualize the complete map after each frame
        plot_data = visualize_new(colors,data,plot_data) 

    disp(plot_data)
    print "Error", np.array(error)/np.array(error_count)
    #if error_count!=1:
    #    print("Error is %f"%(error/(error_count-1)))
    #else:
    #    print "No predictions were made since point was not in view sufficiently"


if __name__ == "__main__":
    test()
