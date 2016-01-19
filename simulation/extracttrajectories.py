import sys
import cPickle
from itertools import izip

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

def rosbag_topic(bagfile, imgtopic):
    bag = rosbag.Bag(bagfile)
    topic, msg, t = bag.read_messages(topics=[imgtopic]).next()
    bridge = CvBridge()
    cvimage = bridge.imgmsg_to_cv2(msg, "bgr8")
    h,w,l = cvimage.shape
    bag = rosbag.Bag(bagfile)

    for topic, msg, t in bag.read_messages(topics=[imgtopic]):
        cvimage = bridge.imgmsg_to_cv2(msg, "bgr8")
        yield t, cvimage

def list_bool_indexing(list_, bool_indices):
    return [e for e, o in izip(list_, bool_indices) if o]

def list_int_indexing(list_, int_indices):
    return [list_[i] for i in int_indices]

def setminus_by_indices(setlist, indices):
    """ A = B \ B[indices] """
    ones = np.ones(len(setlist), dtype=bool)
    ones[indices] = False
    if isinstance(setlist, np.ndarray):
        return setlist[ones]
    else:
        return list_bool_indexing(setlist, ones)



class _Track(object):
    def __init__(self, opencv_keyPoint, time, desc):
        self.kp = [opencv_keyPoint]
        self.ts = [time]
        self.last_desc = desc
        assert(desc.shape == (128,))

    def append(self, opencv_keyPoint, time, desc):
        self.kp.append(opencv_keyPoint)
        self.ts.append(time)
        assert(desc.shape == (128,))
        self.last_desc = desc

class TrackCollection(object):
    def __init__(self):
        # list of continuous tracks
        self._tracks                   = list()

        # if a track with same index is alive
        self._tracks_alive             = list()

    def assert_state(self):
        # assertions to the guarantees of the class
        assert(len(self._tracks_alive) == len(self._tracks))

    def add_new_tracks(self, new_tracks, time, new_desc):
        self._tracks.extend([_Track(t, time, desc) for t, desc in
                             izip(new_tracks, new_desc)])
        self._tracks_alive.extend([1] * len(new_tracks))

        self.assert_state()

    def extend_tracks(self, time, prev_match_indices, matched_points,
                      matched_desc):
        """ 
        prev_match_indices : indices into prev_alive_indices
                             that survived
        """
        # indices into self._tracks that were alive 
        prev_alive_indices = [i for i, a in enumerate(self._tracks_alive) if a]

        # indices into self._tracks that survived
        # survived_track_indices = prev_alive_indices[prev_match_indices]
        survived_track_indices = [
            prev_alive_indices[i] for i in prev_match_indices]

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
        for i in track_indices_that_died:
            self._tracks_alive[i] = False
        self.assert_state()

    def isempty(self):
        return not len(self._tracks)

    def get_alive_tracks(self):
        tracks = list_bool_indexing(self._tracks, self._tracks_alive)
        return tracks

    def tracks(self):
        return self._tracks

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



def main():
    bagfile = sys.argv[1]# "/home/vikasdhi/data/articulatedslam/2016-01-15/all_dynamic_2016-01-15-16-07-26.bag"
    videoout = sys.argv[2]#"/tmp/aae_extracttrajectories%04d.png"
    pickleout = sys.argv[3]#"/tmp/out.pickle"
    imgtopic = '/camera/rgb/image_rect_color'

    prev_img = None
    detector = cv2.SIFT()
    FLANN_INDEX_KDTREE = 0
    matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE,
                                         trees=5),
                                    dict(checks=50))
    tracks = TrackCollection() # 

    prev_img = None
    for timestamp, img in rosbag_topic(bagfile, imgtopic):
        if tracks.isempty():
            kp, desc = detector.detectAndCompute(img, None)
            desc_list = [desc[i, :] for i in xrange(desc.shape[0])]
            tracks.add_new_tracks(kp, timestamp, desc_list)
        else:
            kp, desc_array = detector.detectAndCompute(img, None)
            desc = [desc_array[i, :] for i in xrange(desc_array.shape[0])]

            prev_tracks = tracks.get_alive_tracks()
            prev_kp = [t.kp[-1] for t in prev_tracks]
            prev_desc = np.vstack([t.last_desc for t in prev_tracks])

            # top two matches
            matches = matcher.match(prev_desc, desc_array)
            max_dist = max(img.shape)/10
            matches = [m for m in matches
                       if (np.linalg.norm(np.asarray(prev_kp[m.queryIdx].pt) - 
                                          np.asarray(kp[m.trainIdx].pt)) < 20)]
            prev_match_indices = [m.queryIdx for m in matches]
            match_indices = [m.trainIdx for m in matches]

            tracks.extend_tracks(timestamp, prev_match_indices, 
                                 list_int_indexing(kp, match_indices),
                                 list_int_indexing(desc, match_indices))

            
            new_tracks = setminus_by_indices(kp, match_indices)
            new_desc = setminus_by_indices(desc, match_indices)
            tracks.add_new_tracks(new_tracks, timestamp, new_desc)

            imgdraw = cv2_drawMatches(prev_img, prev_kp, img, kp, matches,
                                      None, matchColor = (0,255,0),
                                      singlePointColor = (255, 0, 0),
                                      matchesMask = None, flags = 0)
            cv2.imshow("c", imgdraw)
            cv2.imwrite(videoout % framecount, imgdraw); framecount += 1
            cv2.waitKey(10)

        prev_img = img

    cPickle.dump(tracks.tracks(), open(pickleout, 'w'))

    
if __name__ == '__main__':
    main()
