import numpy as np
import landmarkmap
import ekf_slam
import cv2

if __name__ == '__main__':
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
                                         scale=10)
    robtraj = landmarkmap.robot_trajectory(np.array([[10, 90], [30,60]]), [10],
                               np.pi/100)

    ldmk_estimater = dict(); # id -> ekf_slam.Estimate_Mm()
    rev_color, pris_color, stat_color = [np.array(l) for l in (
        [255, 0, 0], [0, 255, 0], [0, 0, 255])]
    # angle on both sides of robot dir
    maxangle = 45*np.pi/180
    # max distance in pixels
    maxdist = 120
    # to get the landmarks with ids that are being seen by robot
    for rs, thetas, ids, rob_state, ldmks in landmarkmap.get_robot_observations(
        lmmap, robtraj, maxangle, maxdist, 
                                              # Do not pass visualizer to
                                              # disable visualization
                                              lmvis=None): 
        colors = []
        for r, theta, id in zip(rs, thetas, ids):
            motion_class = ldmk_estimater.setdefault(id, ekf_slam.Estimate_Mm())
            obs = [r, theta]
            motion_class.process_inp_data(obs, rob_state)
            color = tuple(np.int64((motion_class.prior[0]*rev_color 
                     + motion_class.prior[1]*pris_color
                     + motion_class.prior[2]*stat_color)))
            colors.append(color)
        print rs, thetas, ids, rob_state, colors

        posdir = map(np.array, ([rob_state[0], rob_state[1]],
                                [np.cos(rob_state[2]), np.sin(rob_state[2])]))
        robview = landmarkmap.RobotView(posdir[0], posdir[1], maxangle, maxdist)
        img = lmvis.genframe(ldmks, robview, colors=colors)
        img = lmvis.drawrobot(robview, img)
        cv2.imshow(lmvis._name, img)
        cv2.waitKey(lmvis.frame_period)
