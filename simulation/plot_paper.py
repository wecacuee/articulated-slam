'''
Generate plots for the paper
'''
import landmarks_motion_models as joint_mod
import articulation_models as am
import numpy as np
import matplotlib.pyplot as plt
import pdb
import temporal_models as tm
import xlrd
from scipy.integrate import odeint
import ekf_articulated_slam as eslam
import ekf_slam as slam


def deriv(y,t): # Return derivative of the array y
    # Comes from the equation of door as represented on the page
    # https://instruct.math.lsa.umich.edu/lecturedemos/ma216/docs/3_3/
    # 5\ddot{\theta}+22\dot{\theta}+45\theta = 0
    return np.array([y[1],-8.4*y[1]-9*y[0]])

def get_plot_door():
    time = np.linspace(0.0,10.0,1000)
    yinit = np.array([np.pi/2,0])
    y = odeint(deriv,yinit,time)
    plt.figure()
    plt.plot(time,y[:,0])
    plt.show()

# What happens when one tries to estimate configuration and 
# motion parameters jointly
def joint_vs_separate():
    center = np.array([2,2])
    radius = 1
    angles = (1.5*np.pi)*np.array([0.1,1.0/6,0.25,0.5,0.6,0.7,0.9])
    noise_cov = np.diag([0.01,0.01])
    
    num_trials = 1
    error_mat = np.zeros((num_trials,4))
    for trial_num in range(num_trials):
        # For current trial
        # Joint Model estimation
        model1 = joint_mod.Revolute_Landmark(4,noise_cov)
        # Separate Model estimation
        model2 = am.Revolute_Landmark()

        inp_data = np.zeros((angles.shape[0],2))
        for i in range(angles.shape[0]):
            noise_x = np.random.normal(0,0.1,2)
            data = np.array([center[0]+radius*np.cos(angles[i])+noise_x[0],\
                    center[1]+radius*np.sin(angles[i])+noise_x[1]])
            model1.process_inp_data(data)
            model2.process_inp_data(data)
            inp_data[i,:] = data
        error_mat[trial_num,:] = np.array([np.linalg.norm(np.array([model1.model_par[0],\
                model1.model_par[1]])-center),np.linalg.norm(model2.config_pars['center']-center),\
                np.linalg.norm(model1.model_par[2]-radius),\
                np.linalg.norm(model2.config_pars['radius']-radius)])
        #print "Average is ",np.mean(error_mat[:trial_num,:],axis=0)
        # Average is  [ 0.71466052  0.08532083  0.60067241  0.03857038]

    # Plotting estimated circles
    plot_angles = np.linspace(0,2*np.pi,100)
    model1_data = np.zeros((plot_angles.shape[0],2))
    model2_data = np.zeros((plot_angles.shape[0],2))
    true_data = np.zeros((plot_angles.shape[0],2))
    
    for i in range(plot_angles.shape[0]):
        model1_data[i,:] = np.array([model1.model_par[0]+model1.model_par[2]*np.cos(plot_angles[i]),\
                model1.model_par[1]+model1.model_par[2]*np.sin(plot_angles[i])])
        model2_data[i,:] = np.array([model2.config_pars['center'][0]+\
                model2.config_pars['radius']*np.cos(plot_angles[i]),\
                model2.config_pars['center'][1]+\
                model2.config_pars['radius']*np.sin(plot_angles[i])])
        true_data[i,:] = np.array([center[0]+radius*np.cos(plot_angles[i]),\
                center[1]+radius*np.sin(plot_angles[i])])
    
        
    # Plotting the position parameters    
    fig = plt.figure(1)
    plt.rc('text', usetex=True)
    plt.plot(inp_data[:,0],inp_data[:,1],'b+',linewidth=3.0,markersize=10, label='Input Data')
    circ1 = plt.plot(model1_data[:,0],model1_data[:,1],linewidth=3,color='r',\
            marker="s",label = "Joint Model")
    circ2 = plt.plot(model2_data[:,0],model2_data[:,1],linewidth=3,color='g',\
            marker="v",label="Separate Model")
    circ3 = plt.plot(true_data[:,0],true_data[:,1],'k-.',linewidth=3,color='k',\
            label="Ground Truth")
    plt.xlabel(r"X $\rightarrow$",fontsize=15)
    plt.ylabel(r"Y $\rightarrow$",fontsize=15)
    plt.xlim([0.6,3.4])
    plt.ylim([0.6,3.4])
    plt.legend(loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()
    
# Order of motion parameter that should be chosen
def motion_order_plots():
    init_state = np.array([0,0,0])
    dt = 1.0 # Time steps of dt sec
    const_vel = tm.temporal_model(init_state,dt)
    # Reading the data from the excel file
    wb = xlrd.open_workbook('../temp/output_angles_ground_truth.xls')
    sh = wb.sheet_by_index(0)
    num_points = 30
    inp_data = np.zeros(num_points+10)
    lk_prob = np.zeros(num_points+10)
    out_state = np.zeros(num_points+10)
    for i in range(1,num_points+1):
        inp_data[i-1] = sh.cell(0,i).value
    inp_data[-10:] = inp_data[-11]
    num_points = num_points+10
    # Passing this data to the algorithm
    for i in range(inp_data.shape[0]):
        lk_prob[i] = const_vel.process_inp_data(inp_data[i])
        out_state[i] = const_vel.state[0]

    plt.rc('text', usetex=True)
    #plt.subplot(2, 1, 1)
    inp = plt.plot(inp_data,'b+',label='Input Data',linewidth=3.0)
    out = plt.plot(np.arange(const_vel.min_data_samples,num_points),
                   out_state[const_vel.min_data_samples:],
                   'r', label='State Output', linewidth=3.0)
    plt.xlabel(r"Time Steps $\rightarrow$",fontsize=15)
    plt.ylabel(r"Value $\rightarrow$",fontsize=15)
    plt.legend(loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

'''
Plotting SLAM Errors over time
'''
def get_slam_errors():
    (true_robot_states_eslam,eslam_states) = eslam.articulated_slam()
    (etrans_err,erot_err) = slam_metric(true_robot_states_eslam,eslam_states)
    (true_robot_states_slam,slam_states) = slam.slam()
    (trans_err,rot_err) = slam_metric(true_robot_states_slam,slam_states)
    plt.figure(1)
    plt.plot(range(1,len(true_robot_states_eslam)),etrans_err)
    plt.figure(2)
    plt.plot(range(1,len(true_robot_states_slam)),trans_err)
    plt.show()
    pdb.set_trace()
    
# Replicating error measures from 
# http://europa.informatik.uni-freiburg.de/files/kuemmerl09auro.pdf
def slam_metric(true_robot_states,slam_states):
    trans_err_measure = []
    rot_err_measure = []
    for i in range(len(true_robot_states)-1):
        dtrans_gt = np.array(true_robot_states[i+1][0:2])-np.array(true_robot_states[i][0:2])
        dtrans = np.array(slam_states[i+1][0:2])-np.array(slam_states[i][0:2])
        drot_gt = np.array(true_robot_states[i+1][2])-np.array(true_robot_states[i][2])
        drot = np.array(slam_states[i+1][2])-np.array(slam_states[i][2])
        trans_err_measure.append(np.linalg.norm(dtrans_gt-dtrans))
        rot_err_measure.append(abs(np.arctan2(np.sin(drot-drot_gt),np.cos(drot-drot_gt))))
    return (trans_err_measure,rot_err_measure)



if __name__=="__main__":
    #joint_vs_separate()
    #get_plot_door()
    #motion_order_plots()
    get_slam_errors()
