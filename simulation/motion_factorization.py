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
import utils_analysis as ua
import pdb
import sample_shapes
import matplotlib.pyplot as plt
import time

# W matrix from the paper - The dimension of w matrix is 
# 3F x N where F is the number of frames where the data is tracked over and N is the number of feature points that are tracked
def assemble_wmat(track_data,w_mat):
    if w_mat is None:
        w_mat = track_data.transpose()
    else:
        w_mat = np.r_[w_mat,track_data.transpose()]
    return w_mat

'''
Estimate motion and shape using the paper "Shape and Motion from Image Streams under Orthography: a Factorization Method" by Carlo Tomasi and Takeo Kanade
'''
def estimate_motion_shape_kanade(w_mat):
    # Get Registered motion matrix
    w_mat = w_mat-np.mean(w_mat,1)
    # Compute SVD of w_mat
    Umat, Emat , Vmat = np.linalg.svd(w_mat, full_matrices=True)
    # This matrix Emat is supposed to have a rank 3
    if (Emat[2]>1e-5):
        # Its full rank to estimate the rotation matrix
        E_sqrt = np.diag(np.sqrt(Emat[:3]))
        # Equation 6 in the paper
        M_hat = np.dot(Umat[:,:3],E_sqrt)
        S_hat = np.dot(E_sqrt,Vmat[:3,:])
        # Estimating the AA^T matrix using least squares
        lst_sol = estimate_A_kanade(np.array(M_hat))
        # Assembling the matrix AA^T from the least square solution
        A_A_t = np.array([[lst_sol[0][0],lst_sol[0][1],lst_sol[0][2]],[lst_sol[0][1],lst_sol[0][3],lst_sol[0][4]],
            [lst_sol[0][2],lst_sol[0][4],lst_sol[0][5]]])
        # Ideally, check if the matrix is positive definite, assuming it is --> calculate A
        A_mat = np.linalg.cholesky(A_A_t)
        # Getting the shape matrix
        S_mat = np.dot(np.linalg.inv(A_mat),S_hat)
        # Getting the rotation matrix
        M_mat = np.dot(M_hat,A_mat)
        ua.analyze_rotation(M_mat)

'''
Least square solution to estimate A using exuations 9,10,11 from the paper
'''
def estimate_A_kanade(M_hat):
    # Numpy least squares solves ax=b
    a_mat = np.zeros([9*(M_hat.shape[0]/3),6])
    b_mat = np.zeros([9*(M_hat.shape[0]/3),])

    # Constructing array for least square estimation using numpy.linalg.lstsq
    for i in range(M_hat.shape[0]/3):
        # Assemble a_mat
        # 1st row
        a_mat[i*9+0,:] = np.array([M_hat[i*3+0][0]*M_hat[i*3+0][0],2*M_hat[i*3+0][0]*M_hat[i*3+0][1],
            2*M_hat[i*3+0][0]*M_hat[i*3+0][2],M_hat[i*3+0][1]*M_hat[i*3+0][1],
            2*M_hat[i*3+0][1]*M_hat[i*3+0][2], M_hat[i*3+0][2]*M_hat[i*3+0][2]])
        # 2nd row
        a_mat[i*9+1,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+0][0],M_hat[i*3+1][0]*M_hat[i*3+0][1]+M_hat[i*3+1][1]*M_hat[i*3+0][0],
            M_hat[i*3+1][0]*M_hat[i*3+0][2]+M_hat[i*3+1][2]*M_hat[i*3+0][0],M_hat[i*3+1][1]*M_hat[i*3+0][1],
            M_hat[i*3+1][1]*M_hat[i*3+0][2]+M_hat[i*3+1][2]*M_hat[i*3+0][1],M_hat[i*3+1][2]*M_hat[i*3+0][2]])
        # 3rd row
        a_mat[i*9+2,:] = np.array([M_hat[i*3+0][0]*M_hat[i*3+2][0],M_hat[i*3+0][0]*M_hat[i*3+2][1]+M_hat[i*3+0][1]*M_hat[i*3+2][0],
            M_hat[i*3+0][2]*M_hat[i*3+2][0]+M_hat[i*3+0][0]*M_hat[i*3+2][2],M_hat[i*3+0][1]*M_hat[i*3+2][1],
            M_hat[i*3+0][2]*M_hat[i*3+2][1]+M_hat[i*3+0][1]*M_hat[i*3+2][2],M_hat[i*3+0][2]*M_hat[i*3+2][2]])
        # 4th row
        a_mat[i*9+3,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+0][0],M_hat[i*3+1][1]*M_hat[i*3+0][0]+M_hat[i*3+1][0]*M_hat[i*3+0][1],
            M_hat[i*3+1][2]*M_hat[i*3+0][0]+M_hat[i*3+1][0]*M_hat[i*3+0][2],M_hat[i*3+1][1]*M_hat[i*3+0][1],
            M_hat[i*3+1][2]*M_hat[i*3+0][1]+M_hat[i*3+1][1]*M_hat[i*3+0][2],M_hat[i*3+1][2]*M_hat[i*3+0][2]])
        # 5th row
        a_mat[i*9+4,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+1][0],2*M_hat[i*3+1][0]*M_hat[i*3+1][1],
            2*M_hat[i*3+1][0]*M_hat[i*3+1][2],M_hat[i*3+1][1]*M_hat[i*3+1][1],
            2*M_hat[i*3+1][1]*M_hat[i*3+1][2], M_hat[i*3+1][2]*M_hat[i*3+1][2]])
        # 6th row
        a_mat[i*9+5,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+2][0],M_hat[i*3+1][1]*M_hat[i*3+2][0]+M_hat[i*3+1][0]*M_hat[i*3+2][1],
            M_hat[i*3+1][2]*M_hat[i*3+2][0]+M_hat[i*3+1][0]*M_hat[i*3+2][2],M_hat[i*3+1][1]*M_hat[i*3+2][1],
            M_hat[i*3+1][2]*M_hat[i*3+2][1]+M_hat[i*3+1][1]*M_hat[i*3+2][2],M_hat[i*3+1][2]*M_hat[i*3+2][2]])
        # 7th row
        a_mat[i*9+6,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+0][0],M_hat[i*3+2][1]*M_hat[i*3+0][0]+M_hat[i*3+2][0]*M_hat[i*3+0][1],
            M_hat[i*3+2][2]*M_hat[i*3+0][0]+M_hat[i*3+2][0]*M_hat[i*3+0][2], M_hat[i*3+2][1]*M_hat[i*3+0][1],
            M_hat[i*3+2][2]*M_hat[i*3+0][1]+M_hat[i*3+2][1]*M_hat[i*3+0][2], M_hat[i*3+2][2]*M_hat[i*3+0][2]])
        # 8th row
        a_mat[i*9+7,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+1][0],M_hat[i*3+2][1]*M_hat[i*3+1][0]+M_hat[i*3+2][0]*M_hat[i*3+1][1],
            M_hat[i*3+2][2]*M_hat[i*3+1][0]+M_hat[i*3+2][0]*M_hat[i*3+1][2], M_hat[i*3+2][1]*M_hat[i*3+1][1],
            M_hat[i*3+2][2]*M_hat[i*3+1][1]+M_hat[i*3+2][1]*M_hat[i*3+1][2],M_hat[i*3+2][2]*M_hat[i*3+1][2]])
        # 9th row
        a_mat[i*9+8,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+2][0],2*M_hat[i*3+2][0]*M_hat[i*3+2][1],
            2*M_hat[i*3+2][0]*M_hat[i*3+2][2],M_hat[i*3+2][1]*M_hat[i*3+2][1],
            2*M_hat[i*3+2][1]*M_hat[i*3+2][2],M_hat[i*3+2][2]*M_hat[i*3+2][2]])

        # Assemble b_mat
        b_mat[i*9:i*9+9] = np.array([1,0,0,0,1,0,0,0,1])
    # Get least squares solution to the problem
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    # lst_sol[0] is the required matrix, lst_sol[1] is sum of residuals, lst_sol[2] is the rank of coefficient matrix
    lst_sol = np.linalg.lstsq(a_mat,b_mat)
    return lst_sol

''' 
Perform SVD on track matrix and estimate the A matrix using paper ''A Multibody factorization method for independently moving objects, Joao Paulo Costerira, Takeo Kanade"

'''
def estimate_motion_shape_costeria(w_mat):
    # Compute SVD of w_mat
    Umat, Emat , Vmat = np.linalg.svd(w_mat, full_matrices=True)
    # Verify rank of Emat
    if (Emat[3]>1e-5):
        # It's a full rank matrix to estimate rotation and translation
        E_sqrt = np.diag(np.sqrt(Emat[:4]))
        # Equation 6 in the paper
        M_hat = np.dot(Umat[:,:4],E_sqrt)
        S_hat = np.dot(E_sqrt,Vmat[:4,:])
        # Estimating the AA^T matrix using least squares
        lst_sol = estimate_A_costeria(np.array(M_hat))
        # Assembling the matrix AA^T from the least square solution
        A_A_t = np.array([[lst_sol[0][0],lst_sol[0][1],lst_sol[0][2],lst_sol[0][3]],[lst_sol[0][1],lst_sol[0][4],lst_sol[0][5],lst_sol[0][6]],
            [lst_sol[0][2],lst_sol[0][5],lst_sol[0][7],lst_sol[0][8]],[lst_sol[0][3],lst_sol[0][6],lst_sol[0][8],lst_sol[0][9]]])
        # Ideally, check if the matrix is positive definite, assuming it is --> calculate A
        # Carrying out SVD to calculate the matrix A
        U, E, V = np.linalg.svd(A_A_t, full_matrices=True)
        A_R = np.dot(U[:,:3],np.diag(np.sqrt(Emat[:3])))
        # Getting A_t from translation constraint
        A_t = np.dot(np.dot(np.diag(np.sqrt(1/Emat[:4])),np.transpose(Umat[:,:4])),np.mean(w_mat,1))
        A_mat = np.hstack([A_R,A_t])
        # Getting the shape matrix
        S = np.dot(np.linalg.inv(A_mat),S_hat)
        # Getting the motion data
        M = np.dot(M_hat,A_mat)
        # How does one go from here?
        pdb.set_trace()


    else:
        # Its not full rank which might be either due to geometry of the points or the joint type
        print "Something interesting is going on but we cann't exactly say what"
'''
Least square solution to estimate A using exuations 9,10,11 from the paper
'''
def estimate_A_costeria(M_hat):
    # Numpy least squares solves ax=b
    a_mat = np.zeros([9*(M_hat.shape[0]/3),10])
    b_mat = np.zeros([9*(M_hat.shape[0]/3),])

    # Constructing array for least square estimation using numpy.linalg.lstsq
    for i in range(M_hat.shape[0]/3):
        # Assemble a_mat
        # 1st row
        a_mat[i*9+0,:] = np.array([M_hat[i*3+0][0]*M_hat[i*3+0][0],2*M_hat[i*3+0][0]*M_hat[i*3+0][1],
            2*M_hat[i*3+0][0]*M_hat[i*3+0][2],2*M_hat[i*3+0][0]*M_hat[i*3+0][3],M_hat[i*3+0][1]*M_hat[i*3+0][1],
            2*M_hat[i*3+0][1]*M_hat[i*3+0][2], 2*M_hat[i*3+0][1]*M_hat[i*3+0][3],M_hat[i*3+0][2]*M_hat[i*3+0][2],
            2*M_hat[i*3+0][2]*M_hat[i*3+0][3],M_hat[i*3+0][3]*M_hat[i*3+0][3]])
        # 2nd row
        a_mat[i*9+1,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+0][0],M_hat[i*3+1][0]*M_hat[i*3+0][1]+M_hat[i*3+1][1]*M_hat[i*3+0][0],
            M_hat[i*3+1][0]*M_hat[i*3+0][2]+M_hat[i*3+1][2]*M_hat[i*3+0][0],M_hat[i*3+1][0]*M_hat[i*3+0][3]+M_hat[i*3+1][3]*M_hat[i*3+0][0],
            M_hat[i*3+1][1]*M_hat[i*3+0][1],M_hat[i*3+1][1]*M_hat[i*3+0][2]+M_hat[i*3+1][2]*M_hat[i*3+0][1], 
            M_hat[i*3+1][1]*M_hat[i*3+0][3]+ M_hat[i*3+1][3]*M_hat[i*3+0][1],M_hat[i*3+1][2]*M_hat[i*3+0][2],
            M_hat[i*3+1][2]*M_hat[i*3+0][3]+M_hat[i*3+1][3]*M_hat[i*3+0][2],M_hat[i*3+1][3]*M_hat[i*3+0][3]])
        # 3rd row
        a_mat[i*9+2,:] = np.array([M_hat[i*3+0][0]*M_hat[i*3+2][0],M_hat[i*3+0][0]*M_hat[i*3+2][1]+M_hat[i*3+0][1]*M_hat[i*3+2][0],
            M_hat[i*3+0][2]*M_hat[i*3+2][0]+M_hat[i*3+0][0]*M_hat[i*3+2][2],M_hat[i*3+0][3]*M_hat[i*3+2][0]+M_hat[i*3+0][0]*M_hat[i*3+2][3],
            M_hat[i*3+0][1]*M_hat[i*3+2][1],M_hat[i*3+0][2]*M_hat[i*3+2][1]+M_hat[i*3+0][1]*M_hat[i*3+2][2], 
            M_hat[i*3+0][3]*M_hat[i*3+2][1]+ M_hat[i*3+0][1]*M_hat[i*3+2][3],M_hat[i*3+0][2]*M_hat[i*3+2][2],
            M_hat[i*3+0][3]*M_hat[i*3+2][2]+M_hat[i*3+0][2]*M_hat[i*3+2][3],M_hat[i*3+0][3]*M_hat[i*3+2][3]])
        # 4th row
        a_mat[i*9+3,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+0][0],M_hat[i*3+1][1]*M_hat[i*3+0][0]+M_hat[i*3+1][0]*M_hat[i*3+0][1],
            M_hat[i*3+1][2]*M_hat[i*3+0][0]+M_hat[i*3+1][0]*M_hat[i*3+0][2],M_hat[i*3+1][3]*M_hat[i*3+0][0]+M_hat[i*3+1][0]*M_hat[i*3+0][3],
            M_hat[i*3+1][1]*M_hat[i*3+0][1],M_hat[i*3+1][2]*M_hat[i*3+0][1]+M_hat[i*3+1][1]*M_hat[i*3+0][2], 
            M_hat[i*3+1][3]*M_hat[i*3+0][1]+ M_hat[i*3+1][1]*M_hat[i*3+0][3],M_hat[i*3+1][2]*M_hat[i*3+0][2],
            M_hat[i*3+1][3]*M_hat[i*3+0][2]+M_hat[i*3+1][2]*M_hat[i*3+0][3],M_hat[i*3+1][3]*M_hat[i*3+0][3]])
        # 5th row
        a_mat[i*9+4,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+1][0],2*M_hat[i*3+1][0]*M_hat[i*3+1][1],
            2*M_hat[i*3+1][0]*M_hat[i*3+1][2],2*M_hat[i*3+1][0]*M_hat[i*3+1][3],M_hat[i*3+1][1]*M_hat[i*3+1][1],
            2*M_hat[i*3+1][1]*M_hat[i*3+1][2], 2*M_hat[i*3+1][1]*M_hat[i*3+1][3],M_hat[i*3+1][2]*M_hat[i*3+1][2],
            2*M_hat[i*3+1][2]*M_hat[i*3+1][3],M_hat[i*3+1][3]*M_hat[i*3+1][3]])
        # 6th row
        a_mat[i*9+5,:] = np.array([M_hat[i*3+1][0]*M_hat[i*3+2][0],M_hat[i*3+1][1]*M_hat[i*3+2][0]+M_hat[i*3+1][0]*M_hat[i*3+2][1],
            M_hat[i*3+1][2]*M_hat[i*3+2][0]+M_hat[i*3+1][0]*M_hat[i*3+2][2],M_hat[i*3+1][3]*M_hat[i*3+2][0]+M_hat[i*3+1][0]*M_hat[i*3+2][3],
            M_hat[i*3+1][1]*M_hat[i*3+2][1],M_hat[i*3+1][2]*M_hat[i*3+2][1]+M_hat[i*3+1][1]*M_hat[i*3+2][2], 
            M_hat[i*3+1][3]*M_hat[i*3+2][1]+ M_hat[i*3+1][1]*M_hat[i*3+2][3],M_hat[i*3+1][2]*M_hat[i*3+2][2],
            M_hat[i*3+1][3]*M_hat[i*3+2][2]+M_hat[i*3+1][2]*M_hat[i*3+2][3],M_hat[i*3+1][3]*M_hat[i*3+2][3]])
        # 7th row
        a_mat[i*9+6,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+0][0],M_hat[i*3+2][1]*M_hat[i*3+0][0]+M_hat[i*3+2][0]*M_hat[i*3+0][1],
            M_hat[i*3+2][2]*M_hat[i*3+0][0]+M_hat[i*3+2][0]*M_hat[i*3+0][2],M_hat[i*3+2][3]*M_hat[i*3+0][0]+M_hat[i*3+2][0]*M_hat[i*3+0][3],
            M_hat[i*3+2][1]*M_hat[i*3+0][1],M_hat[i*3+2][2]*M_hat[i*3+0][1]+M_hat[i*3+2][1]*M_hat[i*3+0][2], 
            M_hat[i*3+2][3]*M_hat[i*3+0][1]+ M_hat[i*3+2][1]*M_hat[i*3+0][3],M_hat[i*3+2][2]*M_hat[i*3+0][2],
            M_hat[i*3+2][3]*M_hat[i*3+0][2]+M_hat[i*3+2][2]*M_hat[i*3+0][3],M_hat[i*3+2][3]*M_hat[i*3+0][3]])
        # 8th row
        a_mat[i*9+7,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+1][0],M_hat[i*3+2][1]*M_hat[i*3+1][0]+M_hat[i*3+2][0]*M_hat[i*3+1][1],
            M_hat[i*3+2][2]*M_hat[i*3+1][0]+M_hat[i*3+2][0]*M_hat[i*3+1][2],M_hat[i*3+2][3]*M_hat[i*3+1][0]+M_hat[i*3+2][0]*M_hat[i*3+1][3],
            M_hat[i*3+2][1]*M_hat[i*3+1][1],M_hat[i*3+2][2]*M_hat[i*3+1][1]+M_hat[i*3+2][1]*M_hat[i*3+1][2], 
            M_hat[i*3+2][3]*M_hat[i*3+1][1]+ M_hat[i*3+2][1]*M_hat[i*3+1][3],M_hat[i*3+2][2]*M_hat[i*3+1][2],
            M_hat[i*3+2][3]*M_hat[i*3+1][2]+M_hat[i*3+2][2]*M_hat[i*3+1][3],M_hat[i*3+2][3]*M_hat[i*3+1][3]])
        # 9th row
        a_mat[i*9+8,:] = np.array([M_hat[i*3+2][0]*M_hat[i*3+2][0],2*M_hat[i*3+2][0]*M_hat[i*3+2][1],
            2*M_hat[i*3+2][0]*M_hat[i*3+2][2],2*M_hat[i*3+2][0]*M_hat[i*3+2][3],M_hat[i*3+2][1]*M_hat[i*3+2][1],
            2*M_hat[i*3+2][1]*M_hat[i*3+2][2], 2*M_hat[i*3+2][1]*M_hat[i*3+2][3],M_hat[i*3+2][2]*M_hat[i*3+2][2],
            2*M_hat[i*3+2][2]*M_hat[i*3+2][3],M_hat[i*3+2][3]*M_hat[i*3+2][3]])

        # Assemble b_mat
        b_mat[i*9:i*9+9] = np.array([1,0,0,0,1,0,0,0,1])
    # Get least squares solution to the problem
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    # lst_sol[0] is the required matrix, lst_sol[1] is sum of residuals, lst_sol[2] is the rank of coefficient matrix
    lst_sol = np.linalg.lstsq(a_mat,b_mat)
    return lst_sol


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
        fig = None;ax = None
        (fig,ax) = uk.plot_points(sampled_pts,fig,ax)

    # Estimate shape and motion 
    estimate_motion_shape_kanade(w_mat)
    #estimate_motion_shape_costeria(w_mat)


