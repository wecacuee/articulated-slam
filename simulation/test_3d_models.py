""" Test sequence for 3D articulation models
        1. Generate sample point from specific model
        2. Assuming the model is known perform ekf
        3. Observe the predicted motion parameters and ground truth values
        4. Make sure they are close
        5. Add noise and observe outputs """

""" 1. Would possibly need to change in_view function since processing @ every observation is expensive as compared to doing it in bulk
    2. Need to create map generation function
    3. Convert robot state to take from output of map generation
    4. Modify visualization so that landmarks not seen are of a different color"""

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
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch

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

def in_view(rob_state,ldmk_pos,plot_data):
    #pt = np.array([2.0,2.0,0.0])
    pt = ldmk_pos
    #apex = np.array([1.0,5.0,0.0])
    apex = np.array([rob_state[0],rob_state[1],0.0])
    max_dist =5.0 
    phi_view = 90*np.pi/180 # Assumption of robot looking in plane (flat) 
    theta_view = rob_state[2] # robot_state[2] 
    aperture = 45*np.pi/180
    basement = np.array([max_dist*np.cos(theta_view)*np.sin(phi_view),
                  max_dist*np.sin(theta_view)*np.sin(phi_view),
                 max_dist*np.cos(phi_view)])
                  
    basement = basement + np.array(apex)
    #if rob_state[0]==0.0 and rob_state[1]==0.0:
    #    plot_data.append((basement[0],basement[1],basement[2],'k','+'))
    #    plot_data.append((basement[0],basement[1],basement[2]+max_dist,'k','+'))
      
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
    col = ['k','r','y','b','g']
    for i in range(pt.shape[0]):
        X,Y,Z = [pt[i][0],pt[i][1],pt[i][2]]
        if colors[i] is 4:
            plot_data.append((X,Y,Z,col[colors[i]],'^'))
        else:
            plot_data.append((X,Y,Z,col[colors[i]],'o'))
    return plot_data

def disp(plot_data,viewing_cone=False):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    if viewing_cone:
        # Adding base circle
        p = Circle((0.0,0.0),2,fill=False)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p,z=2.0,zdir="y")
        
        datx = [[0.0,0.0],[0.0,0.0],[0.0,-2.0],[0.0,2.0],[0.0,0.0]]
        daty = [[0.0,2.0],[0.0,2.0],[0.0,2.0],[0.0,2.0],[0.0,2.0]]
        datz = [[0.0,0.0],[0.0,2.0],[0.0,0.0],[0.0,0.0],[0.0,-2.0]]
        # Adding lines connecting 4 extreme points to origin
        for it in range(5):
            plt.plot(datx[it],daty[it],datz[it],'k')
    
    for X,Y,Z,ccol,mrk in plot_data:
        ax.scatter(X,Y,Z,c=ccol,marker=mrk)    

    plt.show()

'''
Propagates robot motion with two different models, one for linear velocity
and other one for a combination of linear and rotational velocity
Inputs are: 
Previous robot state,
covarinace in previous state,
actual robot input (translational and rotational component),
and time interval
'''
def robot_motion_prop(prev_state,prev_state_cov,robot_input,delta_t=1):
    # Robot input is [v,w]^T where v is the linear velocity and w is the rotational component
    v = robot_input[0];w=robot_input[1];
    # Robot state is [x,y,\theta]^T, where x,y is the position and \theta is the orientation
    x = prev_state[0]; y=prev_state[1];theta = prev_state[2]
    robot_state = np.zeros(3)
    # Setting noise parameters, following Page 210 Chapter 7, Mobile Robot Localization of 
    alpha_1 = 0.1; alpha_2=0.05; alpha_3 = 0.05; alpha_4 = 0.1
    # M for transferring noise from input space to state-space
    M = np.array([[(alpha_1*(v**2))+(alpha_2*(w**2)),0],[0,(alpha_3*(v**2))+(alpha_4*(w**2))]])
    
    # Check if rotational velocity is close to 0
    if (abs(w)<1e-4):
        robot_state[0] = x+(v*delta_t*np.cos(theta))        
        robot_state[1] = y+(v*delta_t*np.sin(theta))        
        robot_state[2] = theta
        # Derivative of forward dynamics model w.r.t previous robot state
        G = np.array([[1,0,-v*delta_t*np.sin(theta)],\
                [0,1,v*delta_t*np.cos(theta)],\
                [0,0,1]])
        # Derivative of forward dynamics model w.r.t robot input
        V = np.array([[delta_t*np.cos(theta),0],[delta_t*np.sin(theta),0],[0,0]])
    else:
        # We have a non-zero rotation component
        robot_state[0] = x+(((-v/w)*np.sin(theta))+((v/w)*np.sin(theta+w*delta_t)))
        robot_state[1] = y+(((v/w)*np.cos(theta))-((v/w)*np.cos(theta+w*delta_t)))
        robot_state[2] = theta+(w*delta_t)
        G = np.array([[1,0,(v/w)*(-np.cos(theta)+np.cos(theta+w*delta_t))],\
                [0,1,(v/w)*(-np.sin(theta)+np.sin(theta+w*delta_t))],\
                [0,0,1]])
        # Derivative of forward dynamics model w.r.t robot input
        # Page 206, Eq 7.11
        V = np.array([[(-np.sin(theta)+np.sin(theta+w*delta_t))/w,\
                (v*(np.sin(theta)-np.sin(theta+w*delta_t)))/(w**2)+((v*np.cos(theta+w*delta_t)*delta_t)/w)],\
                [(np.cos(theta)-np.cos(theta+w*delta_t))/w,\
                (-v*(np.cos(theta)-np.cos(theta+w*delta_t)))/(w**2)+((v*np.sin(theta+w*delta_t)*delta_t)/w)],\
                [0,delta_t]])
    # Covariance in propagated state
    state_cov = np.dot(np.dot(G,prev_state_cov),np.transpose(G))+np.dot(np.dot(V,M),np.transpose(V))
    return robot_state,state_cov


def test():
    # Initialize required variables
    m_thresh = 0.6

    # Write to file for evaluation
    #f_gt = open('gt_pris.txt','w')
    #f_pred = open('pred_pris.txt','w')


    # Generate map using a single function
    # 
    # Need to modify data generation
    data = gen_cmplx_data('pris','rev')
    data_init_pt = gen_init('pris')
    data2_init_pt = gen_init('rev')   
    ids = gen_id()
    init_pt = gen_init_coll(data_init_pt,data2_init_pt) 
 
    ldmk_estimater = dict(); # id -> mm.Estimate_Mm()
    ldmk_am = dict(); # id->am Id here maps to the chosen Articulated model for the landmark
    ekf_map_id = dict() # Formulating consistent EKF mean and covariance vectors

    # Take robot state from map generation function and add input
    # Case 1: robot at origin
    robot_state_and_input = np.array([0,0,np.pi*90/180])

    # Case 2: Robot away from origin with some theta
    #robot_state = np.array([1,5,np.pi*90/180])
    
    # For visualization
    plot_data =[(robot_state_and_input[0],robot_state_and_input[1],0,'k','+')] 


    slam_state = np.array(robot_state_and_input[:3])
    
    slam_cov = np.diag(np.ones(slam_state.shape[0]))
    ld_ids = []
    index_set = [slam_state.shape[0]]

    Q_obs = np.array([[5.0,0.0,0.0],[0.0,5.0,0.0],[0.0,0.0,5.0]])

    #obs_num = 0
    true_robot_states = []
    slam_robot_states = []


    model = [0,0]
    #error = [0.0,0.0]
    #error_count = [0,0]



    # Noise
    #mu = 0.0
    #sigma = 0.1
    #noise = np.random.normal(mu,sigma,3)
    #print "Noise is :", noise 
    frame = 1 
    true_state = robot_state_and_input[:3]
    for fidx,data,init_point in zip(ids,data,init_pt):
        rob_state = robot_state_and_input[:3]
        robot_input = np.array([0.0,0.0])
        # robot_input = rob_state_and_input[3:]

        #posdir
        #robview
        slam_state[0:3],slam_cov[0:3,0:3] = robot_motion_prop(slam_state[0:3],slam_cov[0:3,0:3],robot_input)
        true_state,_cov = robot_motion_prop(true_state,slam_cov[0:3,0:3],robot_input)
        #plot_data.append((true_state[0],true_state[1],0,'k','+'))


        if len(ld_ids)>0:
            for (ld_id,start_ind,end_ind) in zip(ld_ids,index_set[:-1],index_set[1:]):
                slam_state[start_ind:end_ind] = ldmk_am[ld_id].predict_motion_pars(slam_state[start_ind:end_ind])
                slam_cov[start_ind:end_ind,start_ind:end_ind] = ldmk_am[ld_id].prop_motion_par_cov(slam_cov[start_ind:end_ind,start_ind:end_ind])





        colors = []
        mm_probs = []
        ld_preds = []
        ld_ids_preds = []
        ids_list = []

        for fid,dat,init_pt in itertools.izip(fidx,data,init_point):
            curr_obs = dat
            #curr_obs = curr_obs + noise 
        
            if in_view(true_state,curr_obs,plot_data):
                motion_class = ldmk_estimater.setdefault(fid, mm.Estimate_Mm())

                chosen_am = ldmk_am.setdefault(fid,None)

                motion_class.process_inp_data([0,0],rob_state,curr_obs,init_pt)

                if ldmk_am[fid] is None:
                # Still estimating model
                    if sum(motion_class.prior>m_thresh)>0:
                        model[fid] = np.where(motion_class.prior>m_thresh)[0]
                        ldmk_am[fid] = motion_class.am[model[fid]]
                        ld_ids.append(fid)
                        curr_ld_state = ldmk_am[fid].current_state()
                        curr_ld_cov = ldmk_am[fid].current_cov()
                        index_set.append(index_set[-1]+curr_ld_state.shape[0])
                        slam_state = np.append(slam_state,curr_ld_state)
                        slam_cov = scipy.linalg.block_diag(slam_cov,curr_ld_cov)


                else:

                    curr_ind = ld_ids.index(fid)
                    lk_pred = ldmk_am[fid].predict_model(motion_class.means[model[fid]])
                    #lk_pred = ldmk_am[fid].predict_model(slam_state[index_set[curr_ind]:index_set[curr_ind+1]])
                    ld_preds.append(lk_pred)
                    ld_ids_preds.append(curr_ind)
                    #diff_vec = lk_pred - slam_state[0:2]
                    R_temp = np.array([[np.cos(-slam_state[2]), -np.sin(-slam_state[2]),0],
                         [np.sin(-slam_state[2]), np.cos(-slam_state[2]),0],
                         [0,0,1]])
                    #
                    pos_list = np.ndarray.tolist(slam_state[0:2]) 
                    pos_list.append(0.0)
                    z_pred = R_temp.T.dot(lk_pred)+np.array(pos_list)


                    H_mat = np.zeros((3,index_set[-1]))
                    theta = slam_state[2]
                    H_mat[0,0:3] = np.array([1,0,-np.sin(theta)*curr_obs[0] - np.cos(theta)*curr_obs[1]])
                    H_mat[1,0:3] = np.array([0,1,(np.cos(theta)-np.sin(theta))*curr_obs[0] - (np.cos(theta)+np.sin(theta))*curr_obs[1]])
                    H_mat[2,0:3] = np.array([0,0,(np.sin(theta)+np.cos(theta))*curr_obs[0] + (np.cos(theta)+np.sin(theta))*curr_obs[1]]) 
                    H_mat[:,index_set[curr_ind]:index_set[curr_ind+1]] = np.dot(np.diag(np.ones(3)),ldmk_am[fid].observation_jac(slam_state[index_set[curr_ind]:index_set[curr_ind+1]],init_pt[fid]))

                    inno_cov = np.dot(H_mat,np.dot(slam_cov,H_mat.T))+Q_obs
                    K_mat = np.dot(np.dot(slam_cov,H_mat.T),np.linalg.inv(inno_cov))
                    slam_state = slam_state+np.hstack((np.dot(K_mat,np.vstack(curr_obs-z_pred))))
                    slam_cov = np.dot(np.identity(slam_cov.shape[0])-np.dot(K_mat,H_mat),slam_cov)
                

                    #if fid==1:
                    #    f_gt.write(str(frame)+" "+str(curr_obs[0])+" "+str(curr_obs[1])+" "+str(curr_obs[2])+"\n")
                    #    f_pred.write(str(frame)+" "+str(lk_pred[0])+" "+str(lk_pred[1])+" "+str(lk_pred[2])+"\n")
            
                    # LK_pred comes out in the robot_frame. We need to convert it back to world frame to match the world coordinate observations
            else:
                print "Not in view"


            if in_view(true_state,curr_obs,plot_data):
                mm_probs.append(motion_class.prior)
            
            ### Visualization (Need to add case for not in view landmark)
            track = 1
            if not in_view(true_state,curr_obs,plot_data):
                colors.append(4)
            else:
                for prob in ldmk_estimater[fid].prior[:3]:
                    if prob > m_thresh:
                        colors.append(track)
                        break
                    track = track + 1
                if track == 4:
                    colors.append(0)

        # Visualize the complete map after each frame
        plot_data = visualize_new(colors,data,plot_data) 
        frame = frame + 1
    #f_gt.close()
    #f_pred.close()
    disp(plot_data)
    


if __name__ == "__main__":
    test()
