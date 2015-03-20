'''
Code to sample points from various 3D rigid body, when one samples only points from a plane or line, the resulting motion estimate is degenerate
'''
import numpy as np
import pdb

'''
Given parameters corresponding to a particular shape and a shape type, this code will sample some points on that
Right now, the sampling is manual for testing purposes but maybe in future, we should make it automated
'''
def sample_points(shape_pars,shape_type):
    if shape_type=='ellipse':
        # Sample the extreme points on a elliptical surface
        # takes 3 shape parameters as argument: Length of major axis along x,y and z
        samples = np.matrix([[0,0,0],[shape_pars[0],0,0],[shape_pars[0]/2.0,0,shape_pars[2]/2.0],
            [shape_pars[0]/2.0,0,-shape_pars[2]/2.0],[shape_pars[0]/2.0,shape_pars[1]/2.0,0],
            [shape_pars[0]/2.0,-shape_pars[1]/2.0,0]])
    elif shape_type=='cuboid':
        # Sampling 8 extreme points on the cuboid surface
        samples = np.matrix([[0,0,0],[shape_pars[0],0,0],[shape_pars[0],shape_pars[1],0],
            [0,shape_pars[1],0],[0,0,shape_pars[2]],[shape_pars[0],0,shape_pars[2]],
            [shape_pars[0],shape_pars[1],shape_pars[2]],[0,shape_pars[1],shape_pars[2]]])
        elif shape_type=='cylinder':
            else:
                print "Not a valid shape type"
        print "Allowed shape types are cuboid, ellipse and cylinder"

    return samples




