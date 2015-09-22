import numpy as np
import landmarkmap
import ekf_slam
import cv2

def threeptmap():
    nframes = 10
    map_conf = [dict(ldmks=np.array([[10, 10]]).T,
                     inittheta=0,
                     initpos=[80, 10],
                     deltheta=0,
                     delpos=[0,0]),
                # prismatic
                dict(ldmks=np.array([[10,10]]).T,
                     inittheta=0,
                     initpos=[20,10],
                     deltheta=0,
                     delpos=[5,0]),
                # revolute
                dict(ldmks=np.array([[0,20]]).T,
                     inittheta=np.pi,
                     initpos=[50,40],
                     deltheta=-10*np.pi/180,
                     delpos=[0,0])
               ]
    lmmap = landmarkmap.map_from_conf(map_conf, nframes)
    lmvis = landmarkmap.LandmarksVisualizer([0,0], [100, 100], frame_period=-1,
                                         scale=3)
    robtraj = landmarkmap.robot_trajectory(np.array([[10, 90], [40,60]]),
                                           5, np.pi/100)
    # angle on both sides of robot dir
    maxangle = 45*np.pi/180
    # max distance in pixels
    maxdist = 120

    return nframes, lmmap, lmvis, robtraj, maxangle, maxdist

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "100":
        nframes, lmmap, lmvis, robtraj, maxangle, maxdist = landmarkmap.hundred_ldmk_map(10)
    else:
        nframes, lmmap, lmvis, robtraj, maxangle, maxdist = threeptmap()

    ldmk_estimater = dict(); # id -> ekf_slam.Estimate_Mm()
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    # to get the landmarks with ids that are being seen by robot
    rob_obs_iter = landmarkmap.get_robot_observations(
        lmmap, robtraj, maxangle, maxdist, 
                                              # Do not pass visualizer to
                                              # disable visualization
                                              lmvis=None)
    frame_period = lmvis.frame_period
    for fidx, (rs, thetas, ids, rob_state_and_input, ldmks) in enumerate(rob_obs_iter): 
        rob_state = rob_state_and_input[:3]
        robot_input = rob_state_and_input[3:]
        print '+++++++++++++ fidx = %d +++++++++++' % fidx
        print 'Robot state:', rob_state
        print 'Observations:', zip(rs, thetas)
        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        colors = []
        mm_probs = []
        for r, theta, id in zip(rs, thetas, ids):
            motion_class = ldmk_estimater.setdefault(id, ekf_slam.Estimate_Mm())
            obs = [r, theta]
            motion_class.process_inp_data(obs, rob_state)
            color = np.int64((motion_class.prior[0]*rev_color 
                     + motion_class.prior[1]*pris_color
                     + motion_class.prior[2]*stat_color))
            color = color - np.min(color)
            colors.append(color)
            mm_probs.append(motion_class.prior)

        img = lmvis.genframe(ldmks, robview, colors=colors)
        img = lmvis.drawrobot(robview, img)

        # Draw estimated trajectory
        for id in ids:
            motion_class = ldmk_estimater[id]
            # Plot trajectory
            if motion_class.prior[0] > 0.9:
                # revolute
                center, radius, theta_0, omega = motion_class.mm[0].get_revolute_par()
                #cv2.ellipse(img, centerpt,
                #            np.int64(radius)*lmvis._scale, 0, theta_0, theta_0 + omega*10,
                #           color=rev_color, thickness=1*lmvis._scale)
                for i in range(10):
                    angle = theta_0 + i * omega
                    pt1 = center + radius * np.array([np.cos(angle),
                                               np.sin(angle)])
                    pt1 = tuple(np.int64(pt1)*lmvis._scale)
                    pt2 = center + radius * np.array([np.cos(angle+np.pi/180),
                                               np.sin(angle+np.pi/180)])
                    pt2 = tuple(np.int64(pt2)*lmvis._scale)
                    if np.all(pt1 <= img.shape) and np.all(pt2 <= img.shape):
                        cv2.line(img, pt1, pt2, color=rev_color, thickness=1*lmvis._scale)
            elif motion_class.prior[1] > 0.9:
                # prismatic
                x0, delx = motion_class.mm[1].get_prismatic_par()
                pt1 = tuple(np.int64(x0)*lmvis._scale)
                pt2 = tuple(np.int64(x0+delx*10)*lmvis._scale)
                if np.all(pt1 <= img.shape) and np.all(pt2 <= img.shape):
                    cv2.line(img, pt1, pt2, color=pris_color, thickness=1*lmvis._scale)
        #colors
        print 'motion_class.priors', mm_probs
    
    
        if fidx in [0, 3, 9]:
            filename = '../media/frame%04d.png' % fidx
            print 'Writing to %s' % filename
            cv2.imwrite(filename, img)

        cv2.imshow(lmvis._name, img)
        keyCode = cv2.waitKey(frame_period)
        if keyCode in [1048608, 32]: # space
            frame_period = lmvis.frame_period if frame_period == -1 else -1
        elif keyCode != -1:
            print 'Keycode = %d' % keyCode
