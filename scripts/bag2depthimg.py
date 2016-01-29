import sys
import os
from os.path import join as pjoin, dirname

import cv2
import numpy as np

from extracttrajectories import rosbag_topic, list_get, os_handledirs

if __name__ == '__main__':
    bagfile = list_get(sys.argv, 1,
                       "/home/vikasdhi/mid/articulatedslam/2016-01-22/all_static_2016-01-22-13-49-34.bag")
    outdir = list_get(sys.argv, 2,
                      dirname(bagfile))

    imgtopic = '/camera/rgb/image_rect_color'
    depthtopic = '/camera/depth_registered/image_raw'

    imgframe_fmt = pjoin(outdir, 'img', 'frame%04d.png')
    depthframe_fmt = pjoin(outdir, 'depth', 'frame%04d.np')

    for i, (timestamp, img, depth) in enumerate(
        rosbag_topic(bagfile, imgtopic, depthtopic)):
        cv2.imwrite(os_handledirs(imgframe_fmt % i), img)
        np.save(open(os_handledirs(depthframe_fmt % i), "w"), depth)
