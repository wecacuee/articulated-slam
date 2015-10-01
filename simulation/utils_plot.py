'''
Plotting Utilities
Ellipse plotting adapted from 
http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/
'''

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pdb
import random

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def ellipse_parameters_from_cov(cov, volume):
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 4 * np.sqrt(vals) # This will make +- 2 sigma bounds
    return theta, width, height


def plot_cov_ellipse( pos,cov,obs_num,rob_state,ld_preds,ld_ids_preds, volume=.15, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(111,aspect='equal')

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    theta, width, height = ellipse_parameters_from_cov(cov, volume)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)
    ellip.set_alpha(0.5)
    ellip.set_facecolor((0,1,0))
    ax.add_artist(ellip)
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    # Draw robot's true pose
    plt.plot(rob_state[0],rob_state[1],'r+',linewidth=3.0,markersize=10)
    # Draw predictions of landmark poses
    colors = ['k','y','m']
    for ld,id in zip(ld_preds,ld_ids_preds):
        plt.plot(ld[0],ld[1],color=colors[id],marker='o',linewidth = 3.0)
    plt.savefig("../media/robot_pose_cov"+str(obs_num)+".png")
    plt.cla()

'''
Lets plot mean and covariances of the robot
'''
def slam_cov_plot(slam_state,slam_cov,obs_num,rob_state,ld_preds,ld_ids_preds):
    plot_cov_ellipse(slam_state[0:2],slam_cov[0:2,0:2],obs_num,rob_state,ld_preds,ld_ids_preds)

