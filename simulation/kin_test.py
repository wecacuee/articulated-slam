import numpy as np
import utils_kin as uk
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    # To test the functions in utils_kin.py
    # Definining the articulated body in initial position
    first_joint = uk.Joint('r', np.zeros(3), 1)
    second_joint = uk.Joint('r', np.array([0,0,0]), 1)
    third_joint = uk.Joint('p', np.array([1,0,0,0]), 1)
    chain = uk.JointChain(first_joint, 
            uk.JointChain(second_joint,
                uk.JointChain(third_joint)))
    # Getting all the points on the articulated chain
    sampled_pts, last_origin = chain.articulate(np.zeros(3))
    # Cocatenating data for all the joints
    data = uk.matrix_pts(sampled_pts)
    # Plotting the resulting linkage
    uk.plot_links(sampled_pts,np.zeros(3))

    # Getting data from the second articulation
    first_joint = uk.Joint('r', np.zeros(3), 1)
    second_joint = uk.Joint('r', np.array([0,np.pi/6,0]), 1)
    third_joint = uk.Joint('p', np.array([0,1,0,0.5]), 1)
    chain = uk.JointChain(first_joint, 
            uk.JointChain(second_joint,
                uk.JointChain(third_joint)))
    # Getting all the points on the articulated chain
    sampled_pts, last_origin = chain.articulate(np.zeros(3))
    # Cocatenating data for all the joints
    data_at = uk.matrix_pts(sampled_pts)
    data = np.hstack([data,data_at])

    # Plotting the resulting linkage
    uk.plot_links(sampled_pts,np.zeros(3))

    # Setting the parameters for RANSAC
    n = 3 # minimum number of points needed to estimate the model 
    k = 20 # maximum number of trials 
    t = 5 # minimum support for a model
    tol = 0.05 # tolerance for a datum association with model

    # RANSAC FITS
    colors = ['b','r','g','k','m','y']
    inds = uk.ransac_fit(data,n,k,t,tol)
    # Plot the generatred link
    color_ind = 0
    (fig,ax) = uk.plot_points(data[inds,-3:],[],[],colors[color_ind])
    # Getting the indices not assigned
    indlist = list(xrange(data.shape[0]))
    while (data.shape[0]-len(inds)>t):
        # There might be other models as well
        left_inds = list(set(indlist)-set(inds))
        # Passing the remaining data to RANSAC again
        c_inds = uk.ransac_fit(data[left_inds,:],n,k,t,tol)
        # Verifying if we need to run the RANSAC again
        org_inds =  [left_inds[i] for i in c_inds]
        # Plotting this link as well
        color_ind = color_ind+1
        (fig,ax) = uk.plot_points(data[org_inds,-3:],fig,ax,colors[color_ind])
        # Appending these indices with original indices
        for i in range(len(org_inds)):
            inds.append(org_inds[i])
    
    plt.show(block=True)

