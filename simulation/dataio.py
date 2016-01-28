from collections import deque, namedtuple

TrackedPoint = namedtuple('TrackedPoint', ['track_id', 'pt', 'depth'])

class TimeSeriesSerializer(object):

    def serialize_tracked_pts(self, tracked_pts):
        cols = list()
        npoints = len(tracked_pts)
        cols.append(npoints)
        for (tid, pt, depth) in tracked_pts:
            cols.extend((tid, pt[0], pt[1], depth))

        return "\t".join(map(str, cols))

    def dump(self, file, time_series, timestamps):
        file.write("\t".join(map(str, timestamps)) + "\n")
        for ts_idx, tracked_pts in time_series.iteritems():
            file.write(self.serialize_tracked_pts(tracked_pts))
            file.write("\n")

    def unpack_tracked_pts(self, tracked_pts_str_cols):
        cols = tracked_pts_str_cols
        npoints = int(cols[0])
        tracked_points = list()
        for i in range(npoints):
            tid = int(cols[1+4*i])
            x, y, depth = map(float, cols[4*i+2:4*i+5])
            pt = (x, y)
            tracked_points.append(TrackedPoint(tid, pt, depth))
        return tracked_points


    def load_iter(self, file_but_first_line):
        for ts_idx, line in enumerate(file_but_first_line):
            yield self.unpack_tracked_pts(line.strip().split("\t"))

    def init_load(self, file):
        line = file.readline().strip()
        timestamps = map(int, line.strip().split("\t"))
        return timestamps

    def load(self, file):
        timestamps = self.init_load(file)

        timeseries = dict()
        for tracked_points in self.load_iter(file):
            timeseries[ts_idx] = tracked_points

        return timeseries, timestamps

FeaturesOdomGT = namedtuple('FeaturesOdomGT', ['tracked_points', 'odom',
                                               'gt_pose'])

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
            pos = map(float, cols[:3])
            quat = map(float, cols[3:7])
            linvel = map(float, cols[7:10])
            angvel = map(float, cols[10:13])
            odom = (pos, quat), (linvel, angvel)

            xyz = map(float, cols[13:16])
            abg = map(float, cols[16:19])
            gt_pose = (xyz, abg)

            features = feat_serializer.unpack_tracked_pts(cols[19:])
            yield FeaturesOdomGT(features, odom, gt_pose)
