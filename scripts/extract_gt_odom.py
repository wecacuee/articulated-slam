import sys

from itertools import izip, imap
from collections import deque, namedtuple

import rospy
import rosbag

from extracttrajectories import TimeSeriesSerializer

class ClosestTimeSeriesPoint(object):
    def __init__(self):
        self._time_series_iter = None
        self._t_msg_buffer = deque()

    def set_time_series_iter(self, t_msg_iter):
        self._time_series_iter = t_msg_iter

    def closest_msg(self, ts):
        closest_odom_t, closest_odom_msg  = self._t_msg_buffer.popleft() \
                if len(self._t_msg_buffer) else (rospy.Time(0), None)
        for t, msg in self._time_series_iter:
            if abs(t.to_sec() - ts.to_sec()
                  ) < abs(closest_odom_t.to_sec() - ts.to_sec()):
                closest_odom_msg = msg
                closest_odom_t = t

            if t > ts:
                self._t_msg_buffer.append((t, msg))
                break
        return closest_odom_t, closest_odom_msg

def gt_pose_file(file):
    for line in file:
        cols = map(lambda x: x.strip(), line.strip().split(","))
        ts = rospy.Time(float(cols[0])*1e-6)
        xyz = map(lambda x: 1e-3 * float(x), cols[1:4])
        abg = map(float, cols[4:7])
        yield ts, (xyz, abg)

def odom_msg_to_pose_twist(odom_msg):
    tuplexyz = lambda m: (m.x, m.y, m.z)
    tuplexyzw = lambda m: (m.x, m.y, m.z, m.w)
    return ((tuplexyz(odom_msg.pose.pose.position),
             tuplexyzw(odom_msg.pose.pose.orientation)),
            (tuplexyz(odom_msg.twist.twist.linear),
             tuplexyz(odom_msg.twist.twist.angular),))

FeaturesOdomGT = namedtuple('FeaturesOdomGT', ['tracked_points', 'odom',
                                               'gt_pose'])

def sync_features_odom_gt(feature_time_series, bag_file, robot_gt):
    bag = rosbag.Bag(bag_file)

    closest_odom = ClosestTimeSeriesPoint()
    func = lambda topic_msg_t: (topic_msg_t[2], topic_msg_t[1])
    closest_odom.set_time_series_iter(
        imap(func, bag.read_messages(topics=['/odom'])))

    closest_gt = ClosestTimeSeriesPoint()
    closest_gt.set_time_series_iter(gt_pose_file(open(robot_gt)))

    time_series_reader = TimeSeriesSerializer()
    feature_ts_file = open(feature_time_series)
    timestamps = time_series_reader.init_load(feature_ts_file)
    for ts, tracked_points in izip(timestamps,
                                   time_series_reader.load_iter(feature_ts_file)):
        ts = rospy.Time(ts * 1e-9)
        odom_ts, odom_msg = closest_odom.closest_msg(ts)
        if odom_msg is None:
            raise StopIteration()

        pose_twist_odom = odom_msg_to_pose_twist(odom_msg)

        gt_ts, gt_msg = closest_gt.closest_msg(ts)
        if gt_msg is None:
            raise StopIteration()

        yield FeaturesOdomGT(tracked_points, pose_twist_odom, gt_msg)

class FeaturesOdomGTSerializer(object):
    def dump(self, file, feat_odom_gt_iter, timestamps):
        file.write("\t".join(map(str, timestamps)))
        file.write("\n")
        feat_serializer = TimeSeriesSerializer()
        for feat, odom, gt_pos in feat_odom_gt_iter:
            cols = list() 
            (pos, quat), (linvel, angvel) = odom
            cols.extend(pos)
            cols.extend(quat)
            cols.extend(linvel)
            cols.extend(angvel)

            xyz, abg = gt_pos
            cols.extend(xyz)
            cols.extend(abg)

            feature_str = feat_serializer.serialize_tracked_pts(feat)
            cols.append(feature_str)
            file.write("\t".join(map(str, cols)))
            file.write("\n")

    def load(self, file):
        timestamps = self.init_load(file)

        timeseries = dict()
        for ts_idx,tracked_points in enumerate(self.load_iter(file)):
            timeseries[ts_idx] = tracked_points

        return timeseries, timestamps

    def init_load(self, file):
        line = file.readline().strip()
        timestamps = map(int, line.split("\t"))
        return timestamps

    def load_iter(self, file):
        feat_serializer = TimeSeriesSerializer()
        for line in file:
            cols = line.strip().split("\t")
            pos = cols[:3]
            quat = cols[3:7]
            linvel = cols[7:10]
            angvel = cols[10:13]
            odom = (pos, quat), (linvel, angvel)

            xyz = cols[13:16]
            abg = cols[16:19]
            gt_pose = (xyz, abg)

            features = feat_serializer.unpack_tracked_pts(cols[19:])
            yield FeaturesOdomGT(features, odom, gt_pose)

def main(feature_time_series, bag_file, robot_gt, feat_odom_gt_out):
    time_series_reader = TimeSeriesSerializer()
    feature_ts_file = open(feature_time_series)
    timestamps = time_series_reader.init_load(feature_ts_file)

    fodomgt_serializer = FeaturesOdomGTSerializer()
    fodomgt_serializer.dump(
        feat_odom_gt_out,
        sync_features_odom_gt(feature_time_series, bag_file, robot_gt),
        timestamps)

if __name__ == '__main__':
    feature_time_series = sys.argv[1]
    bag_file = sys.argv[2]
    robot_gt = sys.argv[3]
    feat_odom_gt_out = open(sys.argv[4], "w") if len(sys.argv) >= 5 else sys.stdout
    main(feature_time_series, bag_file, robot_gt, feat_odom_gt_out)
