'''
Implements motion factorization for a single link joint following the paper
''A Multibody factorization method for independently moving objects, Joao Paulo Costerira, Takeo Kanade"

Important Steps to test the algorithm are
1) Defining a articulation joint : Currently supported types are general rigid body motion, revolute, prismatic and motion on a plane
2) Sample data from the motion and assemble w matrix
3) Factorize using SVD and estimate A matrix

'''
import numpy as np
import utils_kin as uk
import pdb
import sample_shapes

# W matrix from the paper - The dimension of w matrix is 
# 3F x N where F is the number of frames where the data is tracked over and N is the number of feature points that are tracked
def assemble_wmat(track_data,w_mat):
    if w_mat is None:
        w_mat = track_data.transpose()
    else:
        w_mat = np.r_[w_mat,track_data.transpose()]
    return w_mat

def estimate_motion_shape(w_mat):
    # Compute SVD of w_mat
    Umat, Emat , Vmat = np.linalg.svd(w_mat, full_matrices=True)
    # Verify rank of Emat
    if (Emat[3]>1e-5):
        # It's a full rank matrix
        E_sqrt = np.diag(np.sqrt(Emat[:4]))
        # Equation 6 in the paper
        M_hat = Umat[:,:4]*E_sqrt
        S_hat = E_sqrt*Vmat[:4,:]
        # Estimating the AA^T matrix

        pdb.set_trace()

    else:
        # Its not full rank which might be either due to geometry of the points or the joint type
        print "Something interesting is going on but we cann't exactly say what"


if __name__ == '__main__':
    # To test the functions in utils_kin.py
    # Definining the articulated body in initial position
    sampled_pts = sample_shapes.sample_points(np.array([1,1,1]),'ellipse')
    first_joint = uk.Joint('f', np.zeros(6), 1,sampled_pts)
    chain = uk.JointChain(first_joint)

    # Initialize w_mat the matrix that is used for factorization of shape and motion
    w_mat = None
    # Pass in a bunch of commands and see how the rotation joint changes things
    # Here first 3 parameters are responsible for rotation and the next 3 for translation
    joint_motion_data = np.array([[0,0,0,0,1,0],[0,np.pi/6,0,1,1,0],[0,np.pi/3,0,0,0,0],[0,np.pi/2,0,0,0,1]])
    origin = np.zeros(3)
    # Processing each motion command
    for curr_motion in joint_motion_data:
        # Changing the motion parameters for the current joint
        chain.joint.jointvars = curr_motion
        # Getting all the points on the articulated chain
        sampled_pts, last_origin = chain.articulate(np.zeros(3))
        data = uk.matrix_pts(sampled_pts)
        print data
        # Collect all the data into factorization matrix
        w_mat = assemble_wmat(data,w_mat)
        # Plotting the resulting linkage
        uk.plot_links(sampled_pts,np.zeros(3),'wire_frame')

    # Estimate shape and motion 
    estimate_motion_shape(w_mat)


