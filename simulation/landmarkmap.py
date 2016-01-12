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
    landmarks = []
    if n >= 4:
        landmarks = [[1,1],
                [maxs[0]-1, 1],
                [maxs[0]-1, maxs[1]-1],
                [1, maxs[1]-1],
                ]
    for i in range(len(landmarks),n):
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
                brea

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

        # Taking dot product for cosine angle
        cosangles = dir.T.dot(cpoints) / dists
        cosangles = cosangles[0, :]

        # The cos angle is negative only when landmark lies behind the robot's heading direction. 
        # Max distance landmarks can be retained. There is no argument or counter-argument yet for in/ex-clusion
        in_view_pts = (cosangles > np.cos(self._maxangle)) & (dists <= self._maxdist)
    
        if len(in_view_pts.shape) > 1:
            import pdb;pdb.set_trace()
        return in_view_pts

def R2D_angle(theta):
    '''
    Return rotation matrix for theta
    '''
    return np.array([[ np.cos(theta),  -np.sin(theta)],
                     [ np.sin(theta),   np.cos(theta)]])

def robot_trajectory(positions, lin_vel, angular_vel):
    '''
    Returns (position_t, direction_t, linear_velocity_{t+1},
    angular_velocity_{t+1})
    '''
    prev_dir = None
    from_pos = positions[:-1]
    to_pos = positions[1:]
    for fp, tp in zip(from_pos, to_pos):
        dir = (tp - fp) / vnorm(tp-fp)

        if prev_dir is not None:
            to_dir = dir
            dir = prev_dir
            # Try rotation, if it increases the projection then we are good.
            after_rot = R2D_angle(angular_vel).dot(prev_dir)
            after_rot_proj = to_dir.dot(after_rot)
            before_rot_proj = to_dir.dot(prev_dir)
            angular_vel = np.sign(after_rot_proj - before_rot_proj) * angular_vel
            # Checks if dir is still on the same side of to_dir as prev_dir
            # Uses the fact that cross product is a measure of sine of
            # differences in orientation. As long as sine of the two
            # differences is same, the product is +ve and the robot must
            # keep rotating otherwise we have rotated too far and we must
            # stop.
            while np.cross(dir, to_dir) * np.cross(prev_dir, to_dir) > 0:
                yield (pos, dir, 0, angular_vel)
                dir = R2D_angle(angular_vel).dot(dir)
            dir = to_dir

        #for i in range(nf+1):
        pos = fp
        vel = (tp - fp) * lin_vel / vnorm(tp - fp)
        # continue till pos is on the same side of tp as fp
        while np.dot((pos - tp), (fp - tp)) > 0:
            yield (pos, dir, lin_vel, 0)
            pos = pos + vel
        prev_dir = dir


class LandmarksVisualizer(object):
    def __init__(self, min, max, frame_period=30, scale=1):
        self._scale = scale
        self._name = "c"
        dims = np.asarray(max) - np.asarray(min)
        nrows = dims[1] * scale
        ncols = dims[0] * scale
        self._imgdims = (nrows, ncols, 3)
        self.frame_period = frame_period
        cv2.namedWindow(self._name, flags=cv2.WINDOW_NORMAL)
        cv2.waitKey(-1)

    def genframe(self, landmarks, robview=None, colors=None):
        img = np.ones(self._imgdims) * 255
        if landmarks.shape[1] > 10:
            radius = 2 * self._scale
        else:
            radius = 4 * self._scale
        red = (0, 0, 255)
        blue = (255, 0, 0)
        black = (0., 0., 0.)
        if robview is not None:
            in_view_ldmks = robview.in_view(landmarks)
        else:
            in_view_ldmks = np.zeros(landmarks.shape[1])

        if colors is not None and len(colors) > 0:
            assert len(colors) == np.sum(in_view_ldmks), '%d <=> %d' % (len(colors), np.sum(in_view_ldmks))
            extcolors = np.empty((landmarks.shape[1], 3))
            extcolors[in_view_ldmks, :] = np.array(colors)
            extcolors[~in_view_ldmks, :] = black
            colors = [tuple(a) for a in list(extcolors)]
        else:
            colors = [(blue if in_view_ldmks[i] else black) for i in
                      range(landmarks.shape[1])]
        for i in range(landmarks.shape[1]):
            pt1 = np.int64(landmarks[:, i]) * self._scale
            if landmarks.shape[1] > 10:
                pt2 = pt1 + radius
                cv2.rectangle(img, tuple(pt1), tuple(pt2), colors[i],
                              thickness=-1)
            else:
                cv2.circle(img, tuple(pt1), radius, colors[i], thickness=-1)
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
        #maxdist = robview._maxdist
        maxdist = 5
        pt1 = np.maximum(np.minimum(self._imgdims[:2], pos.ravel()), [0, 0])
        pt2 = pos + R2D_angle(maxangle).dot(dir) * maxdist
        pt2 = np.maximum(np.minimum(self._imgdims[:2], pt2.ravel()), [0, 0])
        pt3 = pos + R2D_angle(-maxangle).dot(dir) * maxdist
        pt3 = np.maximum(np.minimum(self._imgdims[:2], pt3.ravel()), [0, 0])
        black = (0,0,0)
        red = (0, 0, 255)
        cv2.fillConvexPoly(img, np.int64([[pt1, pt2, pt3]]) * self._scale,
                          color=red)
        cv2.line(img, tuple(np.int64(pt1) * self._scale),
                      tuple(np.int64(pt2) * self._scale), 
                      black)
        cv2.line(img, tuple(np.int64(pt1) * self._scale),
                      tuple(np.int64(pt3) * self._scale), 
                      black)
        cv2.line(img, tuple(np.int64(pt2) * self._scale),
                 tuple(np.int64(pt3) * self._scale), black)
        return img

    def visualizemap_with_robot(self, map, robottraj_iter):
        frame_period = self.frame_period
        for lmks, posdir in itertools.izip(map.get_landmarks(), 
                                           robottraj_iter):
            robview = RobotView(posdir[0], posdir[1], 45*np.pi/180, 40)
            img = self.genframe(lmks, robview)
            img = self.drawrobot(robview, img)
            cv2.imshow(self._name, img)
            if cv2.waitKey(frame_period) >= 0:
                frame_period = self.frame_period if self.frame_period == -1 else -1



def T_from_angle_pos(theta, pos):
    return np.array([[np.cos(theta),  np.sin(theta), pos[0]],
                     [-np.sin(theta), np.cos(theta), pos[1]],
                    [0,            0,               1]])

def get_robot_observations(lmmap, robtraj, maxangle, maxdist, lmvis=None):
    """ Return a tuple of r, theta and ids for each frame"""
    """ v2.0 Return a tuple of lndmks in robot frame and ids for each frame"""
    for ldmks, posdir_and_inputs in itertools.izip(lmmap.get_landmarks(), 
                                       robtraj):
        
        posdir = posdir_and_inputs[:2]
        robot_inputs = posdir_and_inputs[2:]
        robview = RobotView(posdir[0], posdir[1], maxangle, maxdist)
        if lmvis is not None:
            img = lmvis.genframe(ldmks, robview)
            img = lmvis.drawrobot(robview, img)
            cv2.imshow(lmvis._name, img)
            cv2.waitKey(lmvis.frame_period)
        in_view_ldmks = robview.in_view(ldmks)
        selected_ldmks = ldmks[:, in_view_ldmks]
        pos = posdir[0].reshape(2,1)
        
        # v1.0 Need to update after new model has been implemented
        dists = np.sqrt(np.sum((selected_ldmks - pos)**2, 0))
        dir = posdir[1]
        #angles = np.arccos(dir.dot((selected_ldmks - pos))/dists)
        obsvecs = selected_ldmks - pos
        rob_theta = np.arctan2(dir[1], dir[0])
        angles = np.arctan2(obsvecs[1, :], obsvecs[0, :]) - rob_theta
        ldmks_idx = np.where(in_view_ldmks)
        
        # Changed selected_ldmks to robot coordinate frame -> looks like we need to directly send           obsvecs with rotation according to heading
        # v2.0 Rename gen_obs 
        # NOTE: Modify R2D_Angle function based on dimensions of feature space
        ldmk_robot_obs = R2D_angle(rob_theta).dot(obsvecs)

        yield (dists, angles, ldmks_idx[0], [float(pos[0]), float(pos[1]),
                                             rob_theta,
                                             float(robot_inputs[0]),
                                             float(robot_inputs[1])], ldmks,ldmk_robot_obs)

def map_from_conf(map_conf, nframes):
    """ Generate LandmarkMap from configuration """
    rmlist = []
    for rmconf in map_conf:
        if 'nsamples' in rmconf and 'shape' in rmconf:
            ldmks = landmarks_from_rectangle(rmconf['nsamples'], rmconf['shape'])
        elif 'ldmks' in rmconf:
            ldmks = rmconf['ldmks']
        else:
            raise RuntimeException('No way to compute ldmks')


        traj = dyn_trajectory(T_from_angle_pos(rmconf['inittheta'],
                                               rmconf['initpos']),
                              T_from_angle_pos(rmconf['deltheta'],
                                               rmconf['delpos']),
                              nframes)
        rm = RigidMotions(RigidBody2D(ldmks), traj)
        rmlist.append(rm)

    return LandmarkMap(rmlist)

def hundred_ldmk_map(sample_per_block=20):
    nframes = 150
    # The map consists of 5 rectangular rigid bodies with given shape and
    # initial position. Static bodies have deltheta=0, delpos=[0,0]
    map_conf = [dict(nsamples=sample_per_block,
                     shape=[50,50],
                     inittheta=0,
                     initpos=[60, 90],
                     deltheta=0,
                     delpos=[0,0]),
                # prismatic
                dict(nsamples=sample_per_block,
                     shape=[50, 10],
                     inittheta=0,
                     initpos=[10, 80],
                     deltheta=0,
                     delpos=[50./(nframes/2), 0]),
                dict(nsamples=sample_per_block,
                     shape=[50, 50],
                     inittheta=0,
                     initpos=[60, 30],
                     deltheta=0,
                     delpos=[0, 0]),
                # revolute
                dict(nsamples=sample_per_block,
                     shape=[25, 5],
                     inittheta=0,
                     initpos=[35, 30],
                     deltheta=2*np.pi/(nframes/2),
                     delpos=[0, 0]),
                dict(nsamples=sample_per_block,
                     shape=[10, 140],
                     inittheta=0,
                     initpos=[0, 0],
                     deltheta=0,
                     delpos=[0, 0])
               ]

    lmmap = map_from_conf(map_conf, nframes)
    lmv = LandmarksVisualizer([0,0], [110, 140], frame_period=80, scale=3)
    robtraj = robot_trajectory(np.array([[20, 130], [50,100], [35,50]]), 
            5, # frame break points
            np.pi/25) # angular velocity
    # angle on both sides of robot dir
    maxangle = 45*np.pi/180
    # max distance in pixels
    maxdist = 80
    return nframes, lmmap, lmv, robtraj, maxangle, maxdist

if __name__ == '__main__':
    """ Run to see visualization of a dynamic map"""
    nframes, lmmap, lmv, robtraj, maxangle, maxdist = hundred_ldmk_map()
    # to get the landmarks with ids that are being seen by robot
    for r, theta, id, rs in get_robot_observations(lmmap, robtraj, 
                                                   maxangle, maxdist,
                                              # Do not pass visualizer to
                                              # disable visualization
                                              lmv): 
        print r, theta, id, rs
