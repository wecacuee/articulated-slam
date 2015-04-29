"""
Landmark map
"""
import itertools

import numpy as np
import numpy.random as nprnd
from numpy.linalg import norm as vnorm

import cv2


def landmarks_from_rectangle(n, maxs):
    """Generate n landmarks within rectangle"""
    # Some random generator
    assert n >= 4
    landmarks = [[1,1],
                [maxs[0]-1, 1],
                [maxs[0]-1, maxs[1]-1],
                [1, maxs[1]-1],
                ]
    for i in range(4,n):
        landmarks.append(nprnd.rand(2) * maxs)
    return np.array(landmarks).T

def applyT(T, points):
    return T[:2, :2].dot(points) + T[:2, 2:3]

class RigidBody2D(object):
    """ A set of landmarks """
    _id_counter = 0;
    def __init__(self, landmarks):
        self._landmarks = landmarks

    def get_landmarks(self, T):
        return applyT(T, self._landmarks)

class RigidMotions(object):
    """ A pair of rigid body and it's transform trajectory"""
    def __init__(self, rb, trajectory):
        self.rb = rb
        self.trajectory = trajectory

    def get_landmarks(self):
        for T in self.trajectory:
            yield self.rb.get_landmarks(T)

def static_trajectory(Tinit, n):
    """ Static trajectory for n frames """
    for i in xrange(n):
        yield Tinit

def prismatic_trajectory(*args):
    """ Prismatic trajectory for n frames """
    return dyn_trajectory(*args)

def dyn_trajectory(Tinit, delT, n):
    """ Revolute trajectory for n frames """
    Titer = Tinit
    # note that copy is necessary, otherwise delPos is just a view on delT
    # which causes it to change when delT changes
    delPos = delT[:2, 2:3].copy() 
    for i in xrange(n):
        tmp = Titer
        positer = Titer[:2, 2:3]
        # Center of rotation is around around positer (TODO: accept it as
        # paramter)
        delT[:2, 2:3] = delPos -delT[:2, :2].dot(positer) + positer 
        Titer = delT.dot(Titer)
        yield tmp

class LandmarkMap(object):
    """ A dynamic map of landmarks """
    def __init__(self, rigid_motions):
        self._range = (1,1)
        self._rigid_motions = rigid_motions

    def get_landmarks(self):
        while True:
            try:
                landmarks = np.empty((2,0))
                for rm in self._rigid_motions:
                    lmk = rm.get_landmarks().next()
                    landmarks = np.hstack((landmarks, lmk))
                yield landmarks
            except StopIteration:
                break

class RobotView(object):
    """ Describes the cone in view by robot """
    def __init__(self, pos, dir, maxangle, maxdist):
        self._pos = pos.reshape((2,1))
        self._dir = dir.reshape((2,1)) / vnorm(dir)
        self._maxangle = maxangle
        self._maxdist = maxdist

    def in_view(self, points):
        """ Returns true for points that are within view """
        pos = self._pos
        dir = self._dir
        cpoints = points - pos
        dists = np.sqrt(np.sum(cpoints**2, axis=0))
        cosangles = dir.T.dot(cpoints) / dists
        cosangles = cosangles[0, :]
        in_view_pts = (cosangles > np.cos(self._maxangle)) & (dists < self._maxdist)
        if len(in_view_pts.shape) > 1:
            import pdb;pdb.set_trace()
        return in_view_pts

def R2D_angle(theta):
    return np.array([[ np.cos(theta),  np.sin(theta)],
                     [-np.sin(theta),  np.cos(theta)]])

def robot_trajectory(positions, nframes, angular_vel):
    prev_dir = None 
    from_pos = positions[:-1]
    to_pos = positions[1:]
    for fp, tp, nf in zip(from_pos, to_pos, nframes):
        dir = (tp - fp) / vnorm(tp-fp)

        if prev_dir is not None:
            from_dir = prev_dir
            to_dir = dir
            while np.abs(1-dir.dot(to_dir)) < 1e-5:
                dir = R2D_angle(angular_vel).dot(dir)
                yield (pos, dir)

        for i in range(nf+1):
            pos = fp + (tp - fp) * i / nf
            yield (pos, dir)
        prev_dir = dir


class LandmarksVisualizer(object):
    def __init__(self, min, max):
        self._name = "c"
        dims = np.asarray(max) - np.asarray(min)
        nrows = dims[1]
        ncols = dims[0]
        self._imgdims = (nrows, ncols, 3)
        cv2.namedWindow(self._name, flags=cv2.WINDOW_NORMAL)
        cv2.waitKey(-1)

    def genframe(self, landmarks, robview=None):
        img = np.ones(self._imgdims) * 255
        radius = 0
        red = (0, 0, 255)
        blue = (255, 0, 0)
        if robview is not None:
            in_view_ldmks = robview.in_view(landmarks)
        else:
            in_view_ldmks = np.zeros(landmarks.shape[1])
        for i in range(landmarks.shape[1]):
            pt1 = np.int8(landmarks[:, i])
            pt2 = pt1 + radius
            cv2.rectangle(img, tuple(pt1), tuple(pt2),
                          red if in_view_ldmks[i] else blue)
        return img

    def visualizeframe(self, landmarks):
        cv2.imshow(self._name, self.genframe(landmarks))

    def visualizemap(self, map):
        for lmks in map.get_landmarks():
            self.visualizeframe(lmks)

    def drawrobot(self, robview, img):
        pos = robview._pos
        dir = robview._dir
        maxangle = robview._maxangle
        maxdist = robview._maxdist
        pt1 = np.maximum(np.minimum(self._imgdims[:2], pos.ravel()), [0, 0])
        pt2 = pos + R2D_angle(maxangle).dot(dir) * maxdist
        pt2 = np.maximum(np.minimum(self._imgdims[:2], pt2.ravel()), [0, 0])
        cv2.line(img, tuple(np.int8(pt1)),
                      tuple(np.int8(pt2)), 
                      (0, 0, 255))
        pt3 = pos + R2D_angle(-maxangle).dot(dir) * maxdist
        pt3 = np.maximum(np.minimum(self._imgdims[:2], pt3.ravel()), [0, 0])
        cv2.line(img, tuple(np.int8(pt1)),
                      tuple(np.int8(pt3)), 
                      (0, 0, 255))
        return img

    def visualizemap_with_robot(self, map, robottraj_iter):
        for lmks, posdir in itertools.izip(map.get_landmarks(), 
                                           robottraj_iter):
            robview = RobotView(posdir[0], posdir[1], 45*np.pi/180, 40)
            img = self.genframe(lmks, robview)
            img = self.drawrobot(robview, img)
            cv2.imshow(self._name, img)
            cv2.waitKey(30)


def T_from_angle_pos(theta, pos):
    return np.array([[np.cos(theta),  np.sin(theta), pos[0]],
                     [-np.sin(theta), np.cos(theta), pos[1]],
                    [0,            0,               1]])


if __name__ == '__main__':
    """ Run to see visualization of a dynamic map"""
    nframes = 600
    # The map consists of 5 rectangular rigid bodies with given shape and
    # initial position. Static bodies have deltheta=0, delpos=[0,0]
    map_conf = [dict(nsamples=20,
                     shape=[50,50],
                     inittheta=0,
                     initpos=[60, 90],
                     deltheta=0,
                     delpos=[0,0]),
                # prismatic
                dict(nsamples=20,
                     shape=[50, 10],
                     inittheta=0,
                     initpos=[10, 80],
                     deltheta=0,
                     delpos=[50./(nframes/2), 0]),
                dict(nsamples=20,
                     shape=[50, 50],
                     inittheta=0,
                     initpos=[60, 30],
                     deltheta=0,
                     delpos=[0, 0]),
                # revolute
                dict(nsamples=10,
                     shape=[25, 5],
                     inittheta=0,
                     initpos=[35, 30],
                     deltheta=2*np.pi/(nframes/2),
                     delpos=[0, 0]),
                dict(nsamples=30,
                     shape=[10, 140],
                     inittheta=0,
                     initpos=[0, 0],
                     deltheta=0,
                     delpos=[0, 0])
               ]

    rmlist = []
    for rmconf in map_conf:
        ldmks = landmarks_from_rectangle(rmconf['nsamples'], rmconf['shape'])
        traj = dyn_trajectory(T_from_angle_pos(rmconf['inittheta'],
                                               rmconf['initpos']),
                              T_from_angle_pos(rmconf['deltheta'],
                                               rmconf['delpos']),
                              nframes)
        rm = RigidMotions(RigidBody2D(ldmks), traj)
        rmlist.append(rm)

    lmmap = LandmarkMap(rmlist)
    lmv = LandmarksVisualizer([0,0], [110, 140])
    robtraj = robot_trajectory(np.array([[20, 130], [50,100], [35,50]]), [250, 250],
                               np.pi/100)
    lmv.visualizemap_with_robot(lmmap, robtraj)