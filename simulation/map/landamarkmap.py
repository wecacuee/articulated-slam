"""
Landmark map
"""
import numpy as np
import numpy.random as nprnd
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

class Trajectory(object):
    """ A list of transforms """
    def __init__(self, transforms):
        self._transforms = transforms

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

class LandmarksVisualizer(object):
    def __init__(self, min, max):
        self._name = "c"
        dims = np.asarray(max) - np.asarray(min)
        nrows = dims[1]
        ncols = dims[0]
        self._imgdims = (nrows, ncols, 3)
        cv2.namedWindow(self._name, flags=cv2.WINDOW_NORMAL)
        cv2.waitKey(-1)

    def genframe(self, landmarks):
        img = np.ones(self._imgdims) * 255
        radius = 0
        for i in range(landmarks.shape[1]):
            pt1 = np.int8(landmarks[:, i])
            pt2 = pt1 + radius
            cv2.rectangle(img, tuple(pt1), tuple(pt2), (255,0,0))
        return img

    def visualizeframe(self, landmarks):
        cv2.imshow(self._name, self.genframe(landmarks))

    def visualizemap(self, map):
        for lmks in map.get_landmarks():
            self.visualizeframe(lmks)
            cv2.waitKey(30)

def T_from_angle_pos(theta, pos):
    return np.array([[np.cos(theta),  np.sin(theta), pos[0]],
                     [-np.sin(theta), np.cos(theta), pos[1]],
                    [0,            0,               1]])


if __name__ == '__main__':
    """ Run to see visualization of a dynamic map"""
    nframes = 300
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
                     delpos=[50./nframes, 0]),
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
                     deltheta=2*np.pi/nframes,
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
    lmv.visualizemap(lmmap)
