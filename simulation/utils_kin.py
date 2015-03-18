import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.spatial as sp
import random
import collections
import pdb
import time

# Sampling method
def ransac_samples(data,n):
    ''' data stroes all the data in a matrix form and each data
    point is a row vector, n is the number of samples needed'''
    # Getting random indices from shape of the data
    ind = random.sample(xrange(data.shape[0]),n)
    return ind

# Rotation matrix corresponding to an angle about z axis
def rot_matrix(t1,t2,t3):
    # Assume euler angle paramterization (z-y-x) to generate an rotation matrix
    # Inputs are three angles t1,t2,t3 which represent yaw, pitch and roll angles
    Rx =  np.matrix([[1,0,0],[0,np.cos(-t1),-np.sin(-t1)],[0,np.sin(-t1),np.cos(-t1)]])
    Ry =  np.matrix([[np.cos(-t2),0,np.sin(-t2)],[0,1,0],[-np.sin(-t2),0,np.cos(-t2)]])
    Rz =  np.matrix([[np.cos(-t3),-np.sin(-t3),0],[np.sin(-t3),np.cos(-t3),0],[0,0,1]])
    return Rx*Ry*Rz 

'''
Added sampled points to the joint, so that articulate_links uses the sampled points rather than
trying to sample its own points

Following the answer here to redefine the way Joint is handled
http://stackoverflow.com/questions/11351032/named-tuple-and-optional-keyword-arguments
-- Gave up on above because namedtuple is immutable
'''
#Joint = collections.namedtuple('Joint', ['joint_type', 'jointvars', 'length'])
'''
Creating a class for joint type so that we can change the associated sampled_pts data
'''
class Joint:
    def __init__(self,joint_type=None,jointvars=None,length=None,sampled_pts=None):
        self.joint_type = joint_type
        self.jointvars = jointvars
        self.length = length
        self.sampled_pts = sampled_pts

class JointChain:
    def __init__(self, joint, rest_of_chain=None):
        self.joint = joint
        self.next_joint = rest_of_chain

    def add_next_joint(self, joint):
        self.next_joint = JointChain(joint)

    def articulate(self, origin):
        all_sample_ptrs = []
        sample_pts, next_origin = \
                articulated_links(self.joint.joint_type, self.joint.jointvars,
                        self.joint.length,self.joint.sampled_pts, origin)
                # Assign sampled_pts as the sampled points of the joint for future use
        if self.joint.sampled_pts is None:
            '''In case we want to relate the point to the first frame only, else we will need to 
            remove this if condition
            '''
            self.joint.sampled_pts = sample_pts
        # Traversing through the chain
        if self.next_joint is None:
            return [sample_pts], next_origin

        rest_of_sample_ptrs, final_origin = self.next_joint.articulate(next_origin)
        all_sample_ptrs.append(sample_pts)
        all_sample_ptrs.extend(rest_of_sample_ptrs)
        return all_sample_ptrs, final_origin

# Data generation on a articulated link problem
def articulated_links(joint,jointvars,length,sampled_pts,origin):
    # Inputs are joint  - Joint type, jointvars - joint variables
    # length - length of current link, origin - starting point of current
    # link in kinematic chain

    # Defining the geometry of the links

    # Rotating these links by rotation matrix
    if (joint=='r'):
        # Joint is revolute and joint variable is the rotation
        if sampled_pts is None:
            # Generating points on current link
            l1_pt = np.zeros((10*length, 3))
            l1_pt[:,0] = np.linspace(0,length,10*length)
            l1_pt = np.matrix(l1_pt)
        else:
            l1_pt = sampled_pts
        # Rotation matrix
        R = rot_matrix(jointvars[0],jointvars[1],jointvars[2])
        l1_pt = l1_pt*R
    elif (joint=='p'):
        # Joint is prismatic, we need the axes of motion and magnitude of motion
        # jointvars first three variables are axis and fourth is the translation motion
        if sampled_pts is None:
            # Generating points on current link
            l1_pt = np.matrix(np.tile(np.linspace(0,length,10*length),(3,1)))
            l1_pt = l1_pt.transpose()
        else:
            l1_pt = sampled_pts

        norm_vec = np.sqrt(jointvars[0]**2+jointvars[1]**2+jointvars[2]**2)
        dc = np.zeros([3,1])
        # Direction cosines of the prismatic joint
        dc[0] = jointvars[0]/norm_vec
        dc[1] = jointvars[1]/norm_vec
        dc[2] = jointvars[2]/norm_vec
        # Getting the points transformed according to direction cosines
        l1_pt[:,0] = (l1_pt[:,0]+jointvars[3])*dc[0]
        l1_pt[:,1]= (l1_pt[:,1]+jointvars[3])*dc[1]
        l1_pt[:,2] =(l1_pt[:,2]+jointvars[3])*dc[2]
        # Translation outwards
        # For inward translation it would be be 1-k
    
    elif (joint=='f'):
        # Allow full motion range : Combination of translation and rotation
        # Sampling points if points are not sampled 
        if sampled_pts is None:
            # Generating points on current link
            l1_pt = np.zeros((10*length, 3))
            l1_pt[:,0] = np.linspace(0,length,10*length)
            l1_pt = np.matrix(l1_pt)
        else:
            l1_pt = sampled_pts
        # Rotation matrix
        R = rot_matrix(jointvars[0],jointvars[1],jointvars[2])
        # First rotation and then the translation part
        l1_pt = l1_pt*R+jointvars[-3:]

    else:
        # Display error message to specify joint variable
        print "Please pass an appropriate joint type"


    # Adding the origin of previous link to x and y
    l1_pt = l1_pt+origin
    return (l1_pt, l1_pt[-1, :])

# Plot all the points
def plot_points(data,fig = None,ax=None,inp_color = None):
    # Takes as input all the points
    if fig is None:
        fig = plt.figure()
        # If no plot is defined
        ax = fig.gca(projection='3d')
    for i in range(len(data)):
        points = np.asarray(data[i])
        hull = sp.ConvexHull(points)
        # Plotting the points with scatter
        if not inp_color:
            ax.plot_trisurf(points[:,0], points[:,1],points[:,2],triangles=hull.simplices,color = 'b')
        else:
            ax.plot_trisurf(points[:,0], points[:,1],points[:,2],triangles=hull.simplices,color = inp_color)
        # Checking if there are a prismatic joint that needs to be denoted
    # Label the axis
    ax.set_xlabel('X'),ax.set_ylabel('Y'),ax.set_zlabel('Z')
    plt.show()

    return (fig,ax)


# Plot all the links
def plot_links(all_data,origin,plt_option='chain'):
    # Takes as input the rotated matrix data
    fig = plt.figure()
    if (plt_option=='wire_frame'):
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_wireframe(np.array(all_data[0][:,0]),np.array(all_data[0][:,1]),np.array(all_data[0][:,2]))
    else:
        colors = ['g','y','c','m','k']
        # If no plot is defined
        ax = fig.gca(projection='3d')

        for i in range(len(all_data)):
            l1 = np.asarray(all_data[i])
            if (i==0):
                # Origin needs to coincide with the first element
                if not np.all([origin==l1[0,:]]):
                    # Draw a line from origin to current point
                    ax.plot([l1[0,0],origin[0]],[l1[0,1],origin[1]],zs=[l1[0,2],origin[2]],linewidth=2,c = 'r')
            elif (i<len(all_data)):
                # Last point on the link needs to coincide with first point
                l_prev = np.asarray(all_data[i-1])
                if not np.all([l_prev[-1,:]==l1[0,:]]):
                    ax.plot([l_prev[-1,0],l1[0,0]],[l_prev[-1,1],l1[0,1]],zs=[l_prev[-1,2],l1[0,2]],linewidth=2,c = 'r')
            # Plotting the points on a link
            ax.plot(l1[:,0],l1[:,1],zs=l1[:,2],linewidth=3,c = colors[i])

            # Plotting the end point with a scatter
            ax.scatter(l1[-1,0],l1[-1,1],l1[-1,2],'z',50,'b')
            # Checking if there are a prismatic joint that needs to be denoted
    # Label the axis
    ax.set_xlabel('X'),ax.set_ylabel('Y'),ax.set_zlabel('Z')
    # Set fixed axis limits
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.set_zlim([-3,3])
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
    # Python doesn't give transpose of V like it should
    V = V.transpose()
    # Step 4: Calculate matrix X
    X = V*U.transpose()
    # Step 5: Verification
    if ((abs(np.linalg.det(X)+1)<0.0002) and (S[2]==0)):
        status = 1
        V[:,2] = -V[:,2] # Changing the sign of V
    elif ((abs(np.linalg.det(X)+1)<0.0002) and (S[2]>0)):
        status = 0
    else:
        status = 1

    # Getting the actual rotation and translation matrix
    R = V*U.transpose()
    T = (np.mean(pt2,0).transpose())-R*(np.mean(pt1,0).transpose())

    return (status,R,T)

# To verify whether points satisfy a rotation and translation model
def verify_r_t(data,R,T,tol):
    # data contains all the point set data from current and next frame
    # R,T are the estimated rotation and translation vector
    # Tol is the tolerance value for accepting a point
    agg_ind = []
    # Points in first frame
    pt1 = data[:,range(3)]
    # Points in second frame
    pt2 = data[:,range(3,6)]
    # Going through all the datapoints
    for i in range(data.shape[0]):
        val_diff = np.linalg.norm(pt2[i,:].transpose()-R*pt1[i,:].transpose()-T)
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
    pt1 = data[:,range(3)]
    pt2 = data[:,range(3,6)]
    while (count<k):
        # Increasing the trial number
        count = count+1
        # Randomly sample n points from data
        ind = ransac_samples(data,n)
        # Using these indices to learn a model
        status,R,T = estimate_r_t(pt1[ind,:],pt2[ind,:])
        # Checking if rotation and translation was estimated correctly
        if (status):
            # Using the established parameters to find points that satisfy the model
            agg_ind = verify_r_t(data,R,T,tol)
            if (len(agg_ind)>t):
                print "succesful model found"
                print agg_ind
                return agg_ind
        else:
            print "Rotation and translation could not be estimated"

         # Adding all the points from data that work with the same model
    return agg_ind

# Cocatenating all the points together
def matrix_pts(sampled_pts):
    data = sampled_pts[0]
    for i in range(1,len(sampled_pts)):
        data = np.vstack([data,sampled_pts[i]])

    return data


if __name__ == '__main__':
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



