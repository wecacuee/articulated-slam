"""
Landmark map
"""
import itertools

import numpy as np
import numpy.random as nprnd
from numpy.linalg import norm as vnorm
import cv2
import visutils

red = (0, 0, 255)
blue = (255, 0, 0)
black = (0., 0., 0.)

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
    return T[:3, :3].dot(points) + T[:3,3:4]

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
def rev_trajectory(Tinit, delT, n):
    """ Revolute trajectory for n frames """
    Titer = Tinit
    for i in xrange(n):
        Titer[:2, :2] = delT[:2, :2].dot(Titer[:2, :2])
        yield Titer

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
                landmarks = np.empty((3,0))
                for rm in self._rigid_motions:
                    lmk = rm.get_landmarks().next()
                    landmarks = np.hstack((landmarks, lmk))
                yield landmarks
            except StopIteration:
                break

class RobotView(object):
    """ Describes the cone in view by robot """
    def __init__(self, img_shape, K, maxX, 
                image_file_format=None, 
                timestamps_file=None):
        self._imgshape = img_shape
        self.maxX = maxX
        if timestamps_file is not None:
            self._timestamps = dict([(int(line.strip()), i) for i, line in
                                enumerate(open(timestamps_file))])
        else:
            self._timestamps = dict()

        # n_h = [n, -h] notation
        # n_h^\top x_h = 0 is the equation of plane
        # where x_h is in homogeneous coordinates
        # Default equation is for z = 0
        self._robot_plane_w = np.array([[0, 0, 1, 0]]).T
        self._robot_heading_c = np.array([[1, 0, 0]]).T
        self.T_w2c = None
        R_x_view_2_z_view = rodrigues([0, 1, 0], -np.pi/2).dot(
            rodrigues([1, 0, 0], np.pi/2))
        self._K_x_view = K.dot(R_x_view_2_z_view)

        self._image_file_fmt = image_file_format
        self._win_name = "d"
        self._wait_period = 30

    def set_robot_pos_theta(self, pos, theta):
        """
        pos   : 3D position in robot plane
        theta : angle of rotation relative to x-axis with axis of rotation 
        """
        assert pos[2] == 0, "Robot is on Z-axis"
        pos = pos[:2]
        R_c2w = rodrigues([0, 0, 1], theta)
        self.T_w2c = np.eye(4)
        self.T_w2c[:3, :3] = R_c2w.T
        t_c2w = self.robot_plane_basis().dot(pos)
        # t_w2c = - R_c2w.T.dot(t_c2w)
        self.T_w2c[:3, 3] = - R_c2w.T.dot(t_c2w)

    def robot_plane_basis(self):
        b1 = self._robot_heading_c
        plane_normal = self._robot_plane_w[:3, :]
        b2 = np.cross(plane_normal.ravel(),
                      self._robot_heading_c.ravel()).reshape(-1,1)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        return np.hstack((b1 - b1.T.dot(plane_normal), 
                          b2 - b2.T.dot(plane_normal)))

    def pos_dir_z(self):
        plane_normal = self._robot_plane_w[:3]
        T_c2w = np.linalg.inv(self.T_w2c)
        origin = np.zeros((3,1))
        pos = homo2euc(T_c2w.dot(euc2homo(origin)))
        pos_z = pos - pos.T.dot(plane_normal)
        dir_ = self.T_w2c[:3, :3].T.dot(self._robot_heading_c)
        dir_z = dir_ - dir_.T.dot(plane_normal)
        return (pos, dir_z / np.linalg.norm(dir_z))

    def maxangle_z(self):
        Kinv = np.linalg.inv(self._K_x_view)
        u1 = Kinv.dot([self._imgshape[1], 0, 1])
        u1 = u1 / np.linalg.norm(u1)
        u2 = Kinv.dot([0, 0, 1])
        u2 = u2 / np.linalg.norm(u2)
        return np.arccos(u2.T.dot(u1))

    def projected(self, points3D_cam):
        points3D = np.asarray(points3D_cam)
        K = self._K_x_view
        projected = homo2euc(K.dot(points3D))
        return projected

    def projected_world(self, points3D_world):
        points3D = np.asarray(points3D_world)
        T = self.T_w2c
        return self.projected(
            homo2euc(T.dot(euc2homo(points3D))))

    def imgidx_by_timestamp(self, timestamp):
        return self._timestamps.get(timestamp, -1)

    def get_img(self, imgidx=-1):
        if self._image_file_fmt is None or imgidx ==  -1:
            img = np.ones((self._imgshape[0], self._imgshape[1], 3),
                          dtype=np.uint8) * 255
        else:
            img = cv2.imread(self._image_file_fmt % imgidx)
            if img is None:
                raise ValueError("file not found %s" % self._image_file_fmt)

        return img

    def drawlandmarks(self, points3D_cam, imgidx=-1, colors=None):
        img = self.get_img(imgidx)

        if colors is None:
            colors = [(blue if points3D_cam[0, i] < self.maxX else black)
                      for i in range(points3D_cam.shape[1])]

        radius = 0.05
        projected = self.projected(points3D_cam)
        in_view = self.in_view_no_depth(projected) & (
            points3D_cam[0, :] > 0.1)
        projected = projected[:, in_view]
        pt2 = self.projected(points3D_cam + radius)[:, in_view]
        colors = [c for i, c in enumerate(colors) if in_view[i]]

        for i in range(pt2.shape[1]):
            cv2.rectangle(img, 
                          tuple(np.int64(projected[:, i])),
                          tuple(np.int64(pt2[:, i])),
                          colors[i], thickness=-1)
        return img

    def drawtracks(self, ldmktracks_cam, imgidx=-1, colors=None):
        img = self.get_img(imgidx)

        if colors is None:
            colors = [(blue if track[-1].pt3D[0, 0] < self.maxX else black)
                      for track in ldmktracks_cam]

        filter_in_view = lambda proj, pt3D: proj[:, self.in_view_no_depth(proj)
                                    & (pt3D[0, :] > 0.1)]
        proj_tracks = [[filter_in_view(self.projected(pt3D_cam), pt3D_cam)
                       for ts, pt3D_cam in track]
                       for track in ldmktracks_cam]
        filtered_proj_tracks = [[pt2D for pt2D in track if pt2D.shape[1] >= 1]
                                for track in proj_tracks]
        filtered_proj_tracks = [track for track in filtered_proj_tracks
                                if len(track)]
        return visutils.cv2_drawTracks(img,
                                       filtered_proj_tracks,
                                       latestpointcolor=np.asarray(colors),
                                       oldpointcolor=np.asarray(colors)*0.4)

    def drawrevaxis(self, img, center, axis_vec, radius, color, size=1):
        norm = np.linalg.norm 
        pt13D = center + axis_vec * size/ norm(axis_vec)
        pt23D = center - axis_vec * size/ norm(axis_vec)
        pt1 = self.projected_world(pt13D.reshape(-1, 1))
        pt2 = self.projected_world(pt23D.reshape(-1, 1))
        cv2.line(img, tuple(np.int64(pt1)),
                      tuple(np.int64(pt2)), 
                      tuple(color))
        return img

    def visualize(self, img):
        cv2.imshow(self._win_name, img)
        cv2.waitKey(self._wait_period)

    def in_view_no_depth(self, projected):
        in_view = ((projected[0, :] >= 0) 
                   & (projected[0, :] < self._imgshape[1]) 
                   & (projected[1, :] >= 0) 
                   & (projected[1, :] < self._imgshape[0]))
        return in_view

    def in_view(self, points3D_world):
        """ Returns true for points that are within view """
        projected = self.projected_world(points3D_world)
        points3D_cam = homo2euc(self.T_w2c.dot(euc2homo(points3D_world)))
        # within image and closer than maxX
        in_view = (self.in_view_no_depth(projected) 
                   & (points3D_cam[0, :] >= 0.5) 
                   & (points3D_cam[0, :] < self.maxX))
        return in_view

def rotmat_z(theta):
    '''
    Return rotation matrix for theta
    '''
    return np.array([[ np.cos(theta),  -np.sin(theta),0],
                     [ np.sin(theta),   np.cos(theta),0],
                     [0,0,1]])

def robot_trajectory(positions, lin_vel, angular_vel, circle_flag=False,r=20,center=np.array([0,0]),nframes=30):
    '''
    Returns (position_t, direction_t, linear_velocity_{t+1},
    angular_velocity_{t+1})
    '''
    if not circle_flag:
        prev_dir = None
        from_pos = positions[:-1]
        to_pos = positions[1:]
        for fp, tp in zip(from_pos, to_pos):
            fp = np.array(fp)
            tp = np.array(tp)
            dir = (tp - fp) / vnorm(tp-fp)

            if prev_dir is not None:
                to_dir = dir
                dir = prev_dir
                # Try rotation, if it increases the projection then we are good.
                after_rot = rotmat_z(angular_vel).dot(prev_dir)
                after_rot_proj = to_dir.dot(after_rot)
                before_rot_proj = to_dir.dot(prev_dir)
                angular_vel = np.sign(after_rot_proj - before_rot_proj) * angular_vel
                # Checks if dir is still on the same side of to_dir as prev_dir
                # Uses the fact that cross product is a measure of sine of
                # differences in orientation. As long as sine of the two
                # differences is same, the product is +ve and the robot must
                # keep rotating otherwise we have rotated too far and we must
                # stop.
                while np.dot(np.cross(dir, to_dir), np.cross(prev_dir, to_dir)) > 0:
                    yield (pos, dir, 0, angular_vel)
                    dir = rotmat_z(angular_vel).dot(dir)
                dir = to_dir

            #for i in range(nf+1):
            pos = fp
            vel = (tp - fp) * lin_vel / vnorm(tp - fp)
            # continue till pos is on the same side of tp as fp
            while np.dot((pos - tp), (fp - tp)) > 0:
                yield (pos, np.arctan2(dir[1],dir[0]), lin_vel, 0)
                pos = pos + vel
            prev_dir = dir
    else:
        w_v = angular_vel
        theta = 0
        cur = None
        for it in range(nframes):
            to_dir = np.array([center[0]+r*np.cos(theta+it*w_v),
                               center[1]+r*np.sin(theta+it*w_v),0])
            if cur is not None:
                # Assumption of moving in counter clockwise direction
                dir = (to_dir - cur)/vnorm(to_dir-cur)
                yield(to_dir, np.pi/2+ theta+it*w_v, r*w_v, w_v)
            else:
                yield(to_dir, np.pi/2, 0, 0)
            cur = to_dir 
        


def crossmat(e):
    return np.array([[0, -e[2], e[1]],
                     [e[2], 0, -e[0]],
                     [-e[1], e[0], 0]])

def rodrigues(e, t):
    e = np.asarray(e).reshape(3,1)
    return np.cos(t) * np.eye(3) + np.sin(t) * crossmat(e) \
            + (1-np.cos(t)) * e.dot(e.T)

def euc2homo(Vh):
    return np.vstack((Vh, np.ones((1, Vh.shape[1]))))

def homo2euc(V):
    return V[:-1, :] / V[-1, :]

class LandmarksVisualizer(object):
    def __init__(self, min, max, frame_period=30, imgshape=(600, 600)):
        self._name = "c"
        dims = np.asarray(max) - np.asarray(min)
        mean_pos = np.asarray(max) / 2 + np.asarray(min) / 2
        self._imgdims = (imgshape[0], imgshape[1], 3)
        self.frame_period = frame_period
        Z = 5;
        self.set_camera_pos([1, 0, 0, np.pi], 
                            np.array([[mean_pos[0], mean_pos[1], Z*2]]).T)
        f = np.array(imgshape[::-1]) * Z / dims;
        self.camera_K = np.array([[f[0], 0, imgshape[1]/2.],
                                  [0, f[1], imgshape[0]/2.],
                                  [0, 0, 1]])

    def set_camera_pos(self, axis_angle, pos):
        self.camera_R_w2c = rodrigues(np.asarray(axis_angle[:3]), axis_angle[3]).T
        self.camera_t_w2c = - self.camera_R_w2c.dot(pos)

    def projectToImage(self, points3D):
        points3D = np.asarray(points3D).reshape(3, -1)
        K, R, t = (self.camera_K, self.camera_R_w2c, self.camera_t_w2c)
        return homo2euc(K.dot(R.dot(points3D) + t))

    def genframe(self, landmarks, ldmk_robot_obs=[], robview=None, colors=None,SIMULATEDDATA=True):
        img = np.ones(self._imgdims) * 255
        if landmarks.shape[1] > 10:
            radius_euc = 0.05
        else:
            radius_euc = 0.4
        K, t = self.camera_K, self.camera_t_w2c
        radius = np.int64(np.maximum(K[0,0], K[1,1])*radius_euc/t[2])
        red = (0, 0, 255)
        blue = (255, 0, 0)
        black = (0., 0., 0.)
        
        if SIMULATEDDATA:

            if robview is not None:
                in_view_ldmks = robview.in_view(landmarks)
            else:
                in_view_ldmks = np.zeros(landmarks.shape[1])
            in_view_ldmks = np.asarray(in_view_ldmks,dtype='bool')

            if colors is not None and len(colors) > 0:
                all_ldmks = np.hstack((landmarks, ldmk_robot_obs))
                extcolors = np.empty((all_ldmks.shape[1], 3))
                extcolors[in_view_ldmks, :] = np.array(colors)
                extcolors[~in_view_ldmks, :] = black
                colors = [tuple(a) for a in list(extcolors)]
            else:

                colors = [(blue if in_view_ldmks[i] else black) for i in
                          range(landmarks.shape[1])]
        else:

            if robview is not None:
                in_view_ldmks = robview.in_view(landmarks)
            else:
                in_view_ldmks = np.zeros(landmarks.shape[1])

            if colors is not None and len(colors) > 0:
                all_ldmks = np.hstack((landmarks, ldmk_robot_obs))
                extcolors = np.empty((all_ldmks.shape[1], 3))
                extcolors[landmarks.shape[1]:, :] = np.array(colors)
                extcolors[:landmarks.shape[1], :] = black
                colors = [tuple(a) for a in list(extcolors)]
            else:

                colors = [(blue if in_view_ldmks[i] else black) for i in
                          range(landmarks.shape[1])]
            
        proj_ldmks = self.projectToImage(landmarks)
        for i in range(proj_ldmks.shape[1]):
            pt1 = np.int64(proj_ldmks[:, i])
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

    def truncated_direction(self, pt1, dir, maxdist):
        imgshape = np.asarray(self._imgdims[:2]).reshape(-1, 1)
        lambda2_options = np.hstack(((imgshape - pt1) / dir[:2],
                                    -pt1 / dir[:2],
                                     np.array([[maxdist, maxdist]]).T))
        lambda2 = np.vstack(
            (np.min(lambda2_options[0, lambda2_options[0, :] >= 0]),
             np.min(lambda2_options[1, lambda2_options[1, :] >= 0])))
        return pt1 + lambda2 * dir

    def project_direction_shift(self, pos, dir, maxdist):
        pt1 = self.projectToImage(pos)
        pt1_plus_dir1 = self.projectToImage(
            pos +  dir * maxdist)
        dir1 = pt1_plus_dir1 - pt1
        pt2 = self.truncated_direction(pt1, dir1, 1)
        return pt2

    def drawrobot(self, robview, img):
        pos, dir = robview.pos_dir_z()
        maxangle = robview.maxangle_z()
        maxdist = robview.maxX

        pt1 = self.projectToImage(pos).ravel()
        pt2 = self.project_direction_shift(pos,
                                           rotmat_z(-maxangle/2).dot(dir),
                                           maxdist).ravel()
        pt3 = self.project_direction_shift(pos,
                                           rotmat_z(maxangle/2).dot(dir),
                                           maxdist).ravel()
        black = (0,0,0)
        red = (0, 0, 255)
        #cv2.fillConvexPoly(img, np.int64([[pt1, pt2, pt3]]), color=red)
        cv2.line(img, tuple(np.int64(pt1)),
                      tuple(np.int64(pt2)), 
                      black)
        cv2.line(img, tuple(np.int64(pt1)),
                      tuple(np.int64(pt3)), 
                      black)
        cv2.line(img, tuple(np.int64(pt2)),
                 tuple(np.int64(pt3)), black)
        return img

    def drawrevaxis(self, img, center, axis_vec, radius, color, size=1):
        norm = np.linalg.norm 
        pt13D = center + axis_vec * size/ norm(axis_vec)
        pt23D = center - axis_vec * size/ norm(axis_vec)
        pt1 = self.projectToImage(pt13D.reshape(-1, 1))
        pt2 = self.projectToImage(pt23D.reshape(-1, 1))
        cv2.line(img, tuple(np.int64(pt1)),
                      tuple(np.int64(pt2)), 
                      tuple(color))
        return img

    def imshow_and_wait(self, img):
        cv2.imshow(self._name, img/255.)
        keyCode = cv2.waitKey(self.frame_period)
        if keyCode in [1048608, 32]: # space
            # toggle visualization b/w continuity and paused state
            self.frame_period = self.frame_period if self.frame_period == -1 else -1
        elif keyCode != -1:
            print 'Keycode = %d' % keyCode


    def visualizemap_with_robot(self, map, robottraj_iter):
        frame_period = self.frame_period
        for lmks, posdir in itertools.izip(map.get_landmarks(), 
                                           robottraj_iter):
            robview = RobotView(img_shape, K, maxX)
            robview.set_robot_pos_theta(posdir[0], posdir[1])
            img = self.genframe(lmks, robview)
            img = self.drawrobot(robview, img)
            cv2.imshow(self._name, img)
            if cv2.waitKey(frame_period) >= 0:
                frame_period = self.frame_period if self.frame_period == -1 else -1

"""Generating arbitrary 3D RT matrix"""
def T_from_angle_pos(thetas, pos):
    a = thetas[0]
    b = thetas[1]
    c = thetas[2]
    return np.array([[np.cos(b)*np.cos(c),-np.cos(b)*np.sin(c),np.sin(b),pos[0]],
                    [np.cos(a)*np.sin(c)+np.sin(a)*np.sin(b)*np.cos(c),np.cos(a)*np.cos(c)-np.sin(a)*np.sin(b)*np.sin(c),-np.sin(a)*np.cos(b),pos[1]],
                    [np.sin(a)*np.sin(c)-np.cos(a)*np.sin(b)*np.cos(c),np.sin(a)*np.cos(c)+np.cos(a)*np.sin(b)*np.sin(c), np.cos(a)*np.cos(b),pos[2]],
                    [0,0,0,1]])
    #return np.array([[np.cos(theta),  np.sin(theta), pos[0]],
    #                 [-np.sin(theta), np.cos(theta), pos[1]],
    #                [0,            0,               1]])

def get_robot_observations(lmmap, robtraj, maxangle, maxdist, imgshape, K, lmvis=None):
    """ Return a tuple of r, theta and ids for each frame"""
    """ v2.0 Return a ituple of lndmks in robot frame and ids for each frame"""
    prev_theta =  None
    robview = RobotView(imgshape, K, maxdist)
    for ldmks, posdir_and_inputs in itertools.izip(lmmap.get_landmarks(), 
                                       robtraj):
        posdir = posdir_and_inputs[:2]
        robot_inputs = posdir_and_inputs[2:]
        rob_theta = posdir[1]
        #assert(dir[2] == 0)
        #theta = np.arctan2(dir[1], dir[0])
        # Handle in new visualizer
        robview.set_robot_pos_theta(posdir[0], rob_theta)
        if lmvis is not None:
            img = lmvis.genframe(ldmks, robview)
            img = lmvis.drawrobot(robview, img)
            cv2.imshow(lmvis._name, img)
            cv2.waitKey(lmvis.frame_period)
        in_view_ldmks = robview.in_view(ldmks)
        selected_ldmks = ldmks[:, in_view_ldmks]
        pos = posdir[0].reshape(3,1)
        # v1.0 Need to update after new model has been implemented
        #dists = np.sqrt(np.sum((selected_ldmks - pos)**2, 0))
        #dir = posdir[1]
        #angles = np.arccos(dir.dot((selected_ldmks - pos))/dists)
        #obsvecs = selected_ldmks - pos
        #rob_theta = np.arctan2(dir[1], dir[0])
        # Wrap around for theta (to make it continuous)
        #if prev_theta is not None and dis:
        #    if abs(prev_theta - rob_theta) > np.pi/4:
        #        import pdb;pdb.set_trace()
        #        if prev_theta > 0.0:
        #            n_int = np.ceil((((prev_theta /np.pi)-1)/2.0))
        #            rob_theta = (2*n_int+1)*np.pi + rob_theta + np.pi
        #        else:
        #            n_int = np.floor((((prev_theta /np.pi)-1)/2.0))
        #            rob_theta = -(2*n_int+1)*np.pi + np.pi - rob_theta
        #            

        #prev_theta = rob_theta 
        
        #angles = np.arctan2(obsvecs[1, :], obsvecs[0, :]) - rob_theta
        ldmks_idx = np.where(in_view_ldmks)
        
        # Changed selected_ldmks to robot coordinate frame -> looks like we need to directly send           obsvecs with rotation according to heading
        # v2.0 Rename gen_obs
        R_c2w = rodrigues([0, 0, 1], rob_theta)
        R_w2c = R_c2w.T
        ldmk_robot_obs = R_c2w.dot(selected_ldmks-pos)
        # Ignoring robot's Z component
        yield (0, ldmks_idx[0], [float(pos[0]), float(pos[1]),
                                             rob_theta,
                                             float(robot_inputs[0]),
                                             float(robot_inputs[1])], ldmks,
               ldmk_robot_obs)

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
        if rmconf['delthetas'] > 0 and np.all(np.asarray(rmconf['delpos']) == 0):
            traj = rev_trajectory(T_from_angle_pos(rmconf['initthetas'],
                        rmconf['initpos']),
                    T_from_angle_pos(rmconf['delthetas'],
                        rmconf['delpos']),
                    nframes)
        else:
            traj = dyn_trajectory(T_from_angle_pos(rmconf['initthetas'],
                        rmconf['initpos']),
                    T_from_angle_pos(rmconf['delthetas'],
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
    lmv = LandmarksVisualizer([0,0], [110, 140], frame_period=80,
                              imgshape=(420, 330))
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
