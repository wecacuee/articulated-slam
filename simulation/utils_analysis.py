'''
The purpose of the function's here is to analyze shapes and joints from Motion and shape matrices estimated by factorization
'''
import math
import numpy as np
import pdb

'''
Convert rotation matrix to euler angles
Adapted from http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
'''

def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x


def analyze_rotation(M_mat):
    # Analyze data for every frame and comparing it to first frame
    for i in range(1,M_mat.shape[0]/3):
        # Estimated rotation matrix is (relative w.r.t first frame)
        R_rel = M_mat[i*3:(i+1)*3,:]*np.transpose(M_mat[0:3,:])
        # Getting euler angles of this rotation matrix
        (z,y,x) = mat2euler(R_rel)
        print "Rotation w.r.t to the first frame in Euler angles is ",'%.2f' %z,'%.2f'%y,'%.2f'%x
        yield R_rel

'''
Analyzes whether a motion is rotation or full 6-D motion and subsequently prints
rotation and/or translation (in case of full 6-D motion) of all the frames relative to the first frame
Input: Expects the motion matrix as estimated by the factorization process and mean of the measurement matrix
'''
def analyze_motion(M_mat,w_mat_mean):
    # Set printing options
    np.set_printoptions(precision=3)
    # To store the results from the translation estimation part
    trans_est = np.zeros((3,(w_mat_mean.shape[0]/3)-1))
    for i, R_rel in enumerate(analyze_rotation(M_mat)):
        # Removing the part of rotation from the mean track estimates to figure out the rotation and translation part
        trans_est[:,i] = np.squeeze(np.asarray(w_mat_mean[(i+1)*3:(i+2)*3]-np.dot(R_rel,w_mat_mean[0:3])))
        print "Translation w.r.t to the first frame is ",trans_est[:,i]
    # Checking whether the joint can be modeled as a revolute joint only
    if (np.max(np.mean(np.abs(trans_est),1))<1e-2):
        print "The joint is of revolute type"
    else:
        print "The joint is not purely revolute"

    # Also verifying whether the translation is not full rank --> Motion along a line or on a plane
    Umat, Emat , Vmat = np.linalg.svd(trans_est, full_matrices=True)
    if (Emat[2]>1e-4):
        print "Full 3D translation motion"
    elif ((Emat[1]>1e-4) and (Emat[2]<1e-4)) :
        # Motion on a plane, for now just determine normal from two vectors and verify by taking cross product with third translation vector
        # Eventually follow SVD to determine normal from here http://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf
        print "Motion on a plane with normal given by", Umat[:,2]
    else:
        print "Motion along a single axes"



    
        


    

