import sys
import os
import cPickle
from itertools import izip
import subprocess
import glob

import cv2
import rospy
import rosbag
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2

# http://stackoverflow.com/questions/15584608/python-opencv2-cv2-cv-fourcc-not-working-with-videowriterb
try:
    cv2_VideoWriter_fourcc = cv2.VideoWriter_fourc
except AttributeError:
    cv2_VideoWriter_fourcc = cv2.cv.CV_FOURCC

def os_handledirs(filename):
    """ Creates directory for filename """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname): os.makedirs(dirname)
    return filename

def rosbag_topic(bagfile, imgtopic):
    """ Extract opencv image from bagfile """
    bag = rosbag.Bag(bagfile)
    topic, msg, t = bag.read_messages(topics=[imgtopic]).next()
    bridge = CvBridge()
    cvimage = bridge.imgmsg_to_cv2(msg, "bgr8")
    h,w,l = cvimage.shape
    bag = rosbag.Bag(bagfile)

    for topic, msg, t in bag.read_messages(topics=[imgtopic]
                                       #, start_time=rospy.Time(1452892057.27)
                                          ):
        cvimage = bridge.imgmsg_to_cv2(msg, "bgr8")
        yield t, cvimage

def list_bool_indexing(list_, bool_indices):
    """ numpy like list indexing with list of bools """
    return [e for e, o in izip(list_, bool_indices) if o]

def list_int_indexing(list_, int_indices):
    """ numpy like list indexing with list of integers """
    return [list_[i] for i in int_indices]

def setminus_by_indices(B, indices):
    """ return B \ B[indices] """
    ones = np.ones(len(B), dtype=bool)
    ones[indices] = False
    if isinstance(B, np.ndarray):
        return B[ones]
    else:
        return list_bool_indexing(B, ones)

class _Track(object):
    """ Container for list of cv2.KeyPoint and time """
    def __init__(self, opencv_keyPoint, time, desc):
        self.kp = [opencv_keyPoint]
        self.ts = [time]
        self.last_desc = desc
        #assert(desc.shape == (128,))

    def append(self, opencv_keyPoint, time, desc):
        self.kp.append(opencv_keyPoint)
        self.ts.append(time)
        #assert(desc.shape == (128,))
        self.last_desc = desc

class TrackCollection(object):
    """ Container for list of _Track objects while keeping track
    of expired (dead) and alive tracks
    """
    def __init__(self):
        # list of continuous tracks
        self._tracks                   = list()

        ## Bool based indexing requires to loop over all tracks all the time
        ## Int based indexing requires only to loop over alive tracks
        # if a track with same index is alive
        #self._tracks_alive             = list()
        self._alive_track_index        = list()
        self.timestamps               = list()

    def assert_state(self):
        # assertions to the guarantees of the class
        #assert(len(self._tracks_alive) == len(self._tracks))
        pass

    def add_new_tracks(self, new_tracks, time, new_desc):
        print("%d tracks added" % len(new_tracks))
        self.timestamps.append(time)
        self._alive_track_index.extend(
            range(len(self._tracks), len(self._tracks) + len(new_tracks)))
        self._tracks.extend([_Track(t, time, desc) for t, desc in
                             izip(new_tracks, new_desc)])
        #self._tracks_alive.extend([True] * len(new_tracks))

        self.assert_state()

    def extend_tracks(self, time, prev_match_indices, matched_points,
                      matched_desc):
        """ 
        prev_match_indices : indices into prev_alive_indices
                             that survived
        """
        # indices into self._tracks that were alive 
        #prev_alive_indices = [i for i, a in enumerate(self._tracks_alive) if a]
        prev_alive_indices = self._alive_track_index
        #assert(set(prev_alive_indices) == set(self._alive_track_index))

        # indices into self._tracks that survived
        # survived_track_indices = prev_alive_indices[prev_match_indices]
        survived_track_indices = [
            prev_alive_indices[i] for i in prev_match_indices]

        print("%d/%d survived" % (len(survived_track_indices),
                                   len(prev_alive_indices)))
        self._dead_tracks(prev_match_indices, prev_alive_indices)


        for i,kp,desc in izip(survived_track_indices,
                              matched_points,
                              matched_desc):
            self._tracks[i].append(kp, time, desc)

        self.assert_state()

    def _dead_tracks(self, prev_match_indices, prev_alive_indices):
        # indices into self._tracks that just died 
        # track_indices_that_died = (
        #       prev_alive_indices \ prev_alive_indices[prev_match_indices])
        track_indices_that_died = setminus_by_indices(prev_alive_indices,
                                                      prev_match_indices)
        print("%d tracks died" % len(track_indices_that_died))
        for i in track_indices_that_died:
            #self._tracks_alive[i] = False
            self._alive_track_index.remove(i)
        self.assert_state()

    def isempty(self):
        return not len(self._tracks)

    def get_alive_tracks(self):
        #tracks = list_bool_indexing(self._tracks, self._tracks_alive)
        tracks = list_int_indexing(self._tracks, self._alive_track_index)
        #assert(set(tracks) == set(tracks2))
        print("%d/%d alive tracks" % (len(tracks), len(self._tracks)))
        return tracks

    def tracks(self):
        return self._tracks

class TrackCollectionSerializer(object):
    def dump(self, file, trackcollection):
        timestamps = sorted(set(trackcollection.timestamps))
        file.write("\t".join(map(str, timestamps)) + "\n")
        for track in trackcollection.tracks():
            tracklen = len(track.kp)
            trackstartidx = timestamps.index(track.ts[0])
            trackendidx = timestamps.index(track.ts[-1])
            points = list()
            for kp in track.kp:
                points.extend([kp.pt[0], kp.pt[1]])

            cols = [tracklen, trackstartidx, trackendidx] + points
            file.write("\t".join(map(str, cols)) + "\n")

    def load(self, file):
        trackcollection = list()
        line = file.readline()
        timestamps = line.split("\t")
        for line in file:
            cols = line.split("\t")
            tracklen, trackstartidx, trackendidx = map(int, cols[:3])
            points = map(float, cols[3:])
            pointpairs = zip(points[:2*tracklen:2], points[1:2*tracklen:2])
            timelist = timestamps[trackstartidx:trackstartidx + tracklen]
            track = zip(pointpairs, timelist)
            trackcollection.append(track)

        return trackcollection



def cv2_drawMatches(img1, kp1, img2, kp2, matches, outimg,
                    matchColor=(0,255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask = None, flags=0):
    if outimg is None:
        outimg = img1
    
    if matchesMask is None:
        matchesMask = [[0,0]]*len(matches)

    for mat, mask in izip(matches, matchesMask):
        idx1, idx2 = mat.queryIdx, mat.trainIdx
        if mask[0] and mask[1]:
            # both masked
            continue
        elif mask[0] or mask[1]:
            # one of them is masked
            plot_ind = 1 if mask[0] else 0
            kp = [kp1[idx1], kp2[idx2]][plot_ind]
            (x, y) = kp.pt
            cv2.circle(outimg, (int(x), int(y)), 4, singlePointColor, 1)
        else:
            # none masked
            (x1, y1) = kp1[idx1].pt
            (x2, y2) = kp2[idx2].pt

            cv2.circle(outimg, (int(x1), int(y1)), 4, matchColor, 1)
            cv2.circle(outimg, (int(x2), int(y2)), 4, matchColor, 1)
            cv2.line(outimg, (int(x1), int(y1)), (int(x2), int(y2)),
                     matchColor, 1)
    return outimg

def cv2_drawTracks(img, tracks, latestpointcolor, oldpointcolor,
                   plot_last_kp = 40,
                   min_color_gradient = 20
                  ):
    col = np.asarray(oldpointcolor)
    for track in tracks.get_alive_tracks():
        prev_kp = None
        color_kp = col
        colinc = ((np.asarray(latestpointcolor) - col) 
                  / max(min_color_gradient, len(track.kp)))

        # Plot only last 40 key points
        for kp in track.kp[-plot_last_kp:]:
            (x2, y2) = kp.pt
            if prev_kp is not None:
                # none masked
                (x1, y1) = prev_kp.pt
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     color_kp, 2)
            else:
                cv2.circle(img, (int(x2), int(y2)), 2, color_kp, 2)

            prev_kp = kp
            color_kp = color_kp + colinc
        cv2.circle(img, (int(x2), int(y2)), 2, color_kp, 2)

    return img

def list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default

def cv2_setParam(obj, type, name, value):
    if type == cv2.PARAM_ALGORITHM:
        obj.setAlgorithm(name, value)
    elif type in (cv2.PARAM_REAL, cv2.PARAM_FLOAT):
        obj.setDouble(name, value)
    elif type == cv2.PARAM_MAT:
        obj.setMat(name, value)
    elif type == cv2.PARAM_STRING:
        obj.setString(name, value)
    elif type == cv2.PARAM_BOOL:
        obj.setBool(name, value)
    elif type == cv2.PARAM_INT:
        obj.setInt(name, value)
    elif type == cv2.PARAM_MAT_VECTOR:
        obj.setMatVector(name, value)
    else:
        raise RuntimeError("Unknown cv2.type:%d" % type)

class TrajectoryExtractor(object):
    def __init__(self):
        pass

    def init(self, bagfile, imgtopic, feature2d_detector, detector_config,
             feature2d_descriptor, framesout,
             max_dist_as_img_fraction):
        ##
        # Initialization
        ## 

        t, img = rosbag_topic(bagfile, imgtopic).next()
        self.max_dist = max(img.shape) / max_dist_as_img_fraction

        self.detector = cv2.FeatureDetector_create(feature2d_detector)
        for name,value in detector_config[feature2d_detector].items():
            cv2_setParam(self.detector, self.detector.paramType(name),
                         name, value)

        self.descriptor = cv2.DescriptorExtractor_create(feature2d_descriptor)
        FLANN_INDEX_KDTREE = 0
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE,
                                             trees=5),
                                        dict(checks=50))
        self.tracks = TrackCollection() # 


        # clear files
        self.framesout = framesout
        assert("%04d" in framesout)
        for framefile in glob.glob(framesout.replace("%04d", "*")):
            os.remove(framefile)

    def detectAndCompute(self, img):
        # Detect and compute
        kp = self.detector.detect(img, None)
        kp, desc_array = self.descriptor.compute(img, kp)
        desc_list = [desc_array[i, :] for i in xrange(desc_array.shape[0])]
        return kp, desc_list

    def isempty(self):
        return self.tracks.isempty()
    
    def get_prev_keyPoint_desc(self):
        prev_tracks = self.tracks.get_alive_tracks()
        prev_kp = [t.kp[-1] for t in prev_tracks]
        prev_desc = np.vstack([t.last_desc for t in prev_tracks])
        return prev_kp, prev_desc

    def matchKeyPoints(self, kp, desc_list, prev_kp, prev_desc):
        # Matching and filtering based on distance
        matches = self.matcher.match(np.asarray(desc_list, dtype=np.float32),
                                np.asarray(prev_desc, dtype=np.float32))

        is_close_match = lambda m: (
            np.linalg.norm(
            np.asarray(kp[m.queryIdx].pt) -
            np.asarray(prev_kp[m.trainIdx].pt)) < self.max_dist)
        matches = [m for m in matches if is_close_match(m)]
        prev_match_indices = [m.trainIdx for m in matches]
        match_indices = [m.queryIdx for m in matches]
        return prev_match_indices, match_indices

    def updateTracks(self, timestamp, kp, desc_list, prev_match_indices,
                     match_indices):
        # Maintaining tracks database
        if len(match_indices):
            self.tracks.extend_tracks(timestamp, prev_match_indices, 
                                 list_int_indexing(kp, match_indices),
                                 list_int_indexing(desc_list, match_indices))

        
        new_tracks = setminus_by_indices(kp, match_indices)
        new_desc = setminus_by_indices(desc_list, match_indices)
        self.tracks.add_new_tracks(new_tracks, timestamp, new_desc)

    def visualize(self, img, framecount):
        ##
        # Visualization
        ##

        # imgdraw = cv2_drawMatches(prev_img, prev_kp, img, kp, matches,
        #                           None, matchColor = (0,255,0),
        #                           singlePointColor = (255, 0, 0),
        #                           matchesMask = None, flags = 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        imgdraw = cv2_drawTracks(img, self.tracks,
                                 latestpointcolor = green,
                                 oldpointcolor = red)
        cv2.imshow("c", imgdraw)
        cv2.imwrite(os_handledirs(self.framesout % framecount), imgdraw)
        cv2.waitKey(10)

def main():
    ##
    # Config
    ## 
    bagfile = list_get(sys.argv, 1,
                       "/home/vikasdhi/data/articulatedslam/2016-01-15/all_dynamic_2016-01-15-16-07-26.bag")
    videoout_temp = list_get(sys.argv, 2, "/tmp/%s_%s_out.avi")
    pickleout_temp = list_get(sys.argv, 3, "/tmp/%s_%s_out.pickle")
    imgtopic = '/camera/rgb/image_rect_color'
    framesout = "/tmp/aae_extracttrajectories/frames%04d.png"
    # maximum distance between matching feature points
    max_dist_as_img_fraction = 20 
    ## The following detector types are supported: 
    # "FAST"  FastFeatureDetector
    # "STAR"  StarFeatureDetector
    # "SIFT"  SIFT (nonfree module)
    # "SURF"  SURF (nonfree module)
    # "ORB" ORB
    # "BRISK" BRISK
    # "MSER" MSER
    # "GFTT" GoodFeaturesToTrackDetector
    # "HARRIS" GoodFeaturesToTrackDetector with Harris detector enabled
    # "Dense" DenseFeatureDetector
    # "SimpleBlob" SimpleBlobDetector
    feature2d_detector = "GFTT"
    # Detector config
    detector_config = dict(
        FAST=dict(threshold=30.0),
        Dense=dict(initXyStep=10.0),
        GFTT=dict(minDistance=0,qualityLevel=0.02),
    )

    ## The following descriptor types are supported: 
    #   "SIFT" SIFT
    #   "SURF" SURF
    #   "BRIEF" BriefDescriptorExtractor
    #   "BRISK" BRISK
    #   "ORB" ORB
    #   "FREAK" FREAK
    feature2d_descriptor = "SIFT"
    videoout = videoout_temp % (feature2d_detector, feature2d_descriptor)
    pickleout = pickleout_temp % (feature2d_detector, feature2d_descriptor)



    traj_extract = TrajectoryExtractor()
    traj_extract.init(bagfile, imgtopic, feature2d_detector, detector_config,
                      feature2d_descriptor, framesout,
                      max_dist_as_img_fraction)
    framecount = 0
    for timestamp, img in rosbag_topic(bagfile, imgtopic):
        kp, desc_list = traj_extract.detectAndCompute(img)
        if traj_extract.isempty():
            traj_extract.updateTracks(timestamp, kp, desc_list, [], [])
        else:
            prev_kp, prev_desc = traj_extract.get_prev_keyPoint_desc()
            prev_match_indices, match_indices = \
                    traj_extract.matchKeyPoints(kp, desc_list, prev_kp,
                                                prev_desc)
            traj_extract.updateTracks(timestamp, kp, desc_list,
                                      prev_match_indices, match_indices)
            traj_extract.visualize(img, framecount)
            framecount += 1


    # Saving tracks
    TrackCollectionSerializer().dump(open(pickleout, 'w'), traj_extract.tracks)
    # Video from frames
    subprocess.call(("""avconv -framerate 12 -i %s -r
                    30 -vb 2M %s""" % (framesout, videoout)).split())
    
if __name__ == '__main__':
    main()
