import matplotlib.pyplot as plt
import numpy as np
import random
import pdb

# Sampling method
def ransac_samples(data,n):
    ''' data stroes all the data in a matrix form and each data
    point is a row vector, n is the number of samples needed'''
    # Getting random indices from shape of the data
    ind = random.sample(xrange(data.shape[0]),n)
    return ind

# Rotation matrix corresponding to an angle about z axis
def rot_matrix(t1): 
    return np.matrix([[np.cos(-t1),-np.sin(-t1)],[np.sin(-t1),np.cos(-t1)]])

# Data generation on a 2 link problem
def articulated_links(theta):
    # Defining the geometry of the links
    l1 = 3;l2 = 1
    # Creating points on the link1 for the rotation
    l1_pt = np.matrix([np.linspace(0,l1,10*l1),np.zeros(10*l1)])
    # Creating a row vectors out of it
    l1_pt = l1_pt.transpose()
    # Creating points on the link1 for the rotation
    l2_pt = np.matrix([np.linspace(0,l2,10*l2),np.zeros(10*l2)])
    # Creating a row vectors out of it
    l2_pt = l2_pt.transpose()
    
    # Rotating these links by rotation matrix
    l1_pt = l1_pt*rot_matrix(theta[0])
    l2_pt = l2_pt*rot_matrix(theta[1])
    # Adding the origin of previous link to x and y
    l2_pt = l2_pt+l1_pt[-1,:]
    return (l1_pt,l2_pt)

# Plot the links
def plot_links(data):
    # Takes as input the rotated matrix data
    l1 = data[0]
    l2 = data[1]
    plt.plot(l1[:,0],l1[:,1])
    plt.hold(True)
    plt.plot(l2[:,0],l2[:,1])
    plt.show()

# Rotation and translation estimation of point sets using the paper
# Least squares fitting of two 3-D points sets by K.S. Arun,Huang and blostein
def estimate_r_t(pt1,pt2):
    # Pt1 is the points in the previous frames and pt2 in current frame

    R = np.zeros([pt1.shape[1],pt1.shape[1]])
    T = np.zeros(pt1.shape[1])
    # Estimating means and subtracting from the original points
    tf_pt1 = pt1-np.mean(pt1,0)
    tf_pt2 = pt2-np.mean(pt2,0)
    # Step 2: Calculating the matrix 
    H = tf_pt1.transpose()*tf_pt2
    # Step 3: Calculate SVD of the matrix
    U,S,V = np.linalg.svd(H,full_matrices=True)
    # Step 4: Calculate matrix X
    X = V*U.transpose()
    # Step 5: Verification
    if (round(np.linalg.det(X))==-1):
        status = 0
    else:
        status = 1
        R = X
        T = np.mean(pt2,0)-np.mean(pt1,0)*R

    return (status,R,T)

# To verify whether points satisfy a rotation and translation model
def verify_r_t(data,R,T,tol):
    # data contains all the point set data from current and next frame
    # R,T are the estimated rotation and translation vector
    # Tol is the tolerance value for accepting a point
    agg_ind = []
    # Points in first frame
    pt1 = data[:,range(2)]
    # Points in second frame
    pt2 = data[:,range(2,4)]
    # Going through all the datapoints
    for i in range(data.shape[0]):
        #pdb.set_trace()
        val_diff = np.linalg.norm(pt2[i,:]-pt1[i,:]*R-T)
        if (val_diff<=tol):
            agg_ind.append(i)
    return agg_ind

# RANSAC Estimation
def ransac_fit(data,n,k,t,tol):
    ''' Implementation of paper Random Sample Consensus: A paradigm for model fitting
    with applications to image analysis and automated cartography'''
    # data is input data and n is the minimum number of points needed for a model
    # k is the maximum number of trials and t is the lower bound of an acceptable
    # consensus set size
    count = 0
    agg_ind = []
    # Splitting into points
    pt1 = data[:,range(2)]
    pt2 = data[:,range(2,4)]
    while (count<k):
        # Increasing the trial number
        count = count+1
        # Randomly sample n points from data
        ind = ransac_samples(data,n)
        # Using these indices to learn a model
        #pdb.set_trace()
        status,R,T = estimate_r_t(pt1[ind,:],pt2[ind,:])
        # Checking if rotation and translation was estimated correctly
        if (status):
            # Using the established parameters to find points that satisfy the model
            agg_ind = verify_r_t(data,R,T,tol)
            if (len(agg_ind)>t):
                print "succesful model found"
                return agg_ind
        else:
            print "Rotation and translation could not be estimated"
         
         # Adding all the points from data that work with the same model
    return agg_ind



# Main script
# Create initial points on the links 
theta = np.array([0,0])
l1_pt,l2_pt = articulated_links(theta)

# Rotating the linkage
theta = np.array([np.pi/6,np.pi/2])
l1_tf,l2_tf = articulated_links(theta)

# Appending all the data
data = np.vstack((np.hstack((l1_pt,l1_tf)),np.hstack((l2_pt,l2_tf))))

# Setting the parameters for RANSAC
n = 3 # minimum number of parameters for model 
k = 10 # maximum number of trials 
t = 7 # minimum support for a model
tol = 0.05 # tolerance for a datum association with model
inds = ransac_fit(data,n,k,t,tol)
print inds

# Verifying if we need to run the RANSAC again
if (data.shape[0]-len(inds)>t):
    # There might be other models as well

    # Getting the indices not assigned
    indlist = list(xrange(data.shape[0]))
    left_inds = list(set(indlist)-set(inds))
    # Passing the remaining data to RANSAC again
    c_inds = ransac_fit(data[left_inds,:],n,k,t,tol)
    org_inds =  [left_inds[i] for i in c_inds]
    print org_inds
    # Displaying these points
    plt.plot(data[inds,2],data[inds,3],'ro')
    plt.hold(True)
    plt.plot(data[org_inds,2],data[org_inds,3],'b+')
    plt.show()
else:
    plt.plot(data[inds,2],data[inds,3],'ro')
    plt.show()



