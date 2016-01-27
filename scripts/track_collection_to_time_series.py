from extracttrajectories import (TrackCollectionSerializer,
                                 TimeSeriesSerializer, 
                                 reindex_as_timeseries)

if __name__ == '__main__':
    import sys
    [track_collection, timestamps] = TrackCollectionSerializer().load(sys.stdin)
    [time_series, time_stamps] = reindex_as_timeseries(track_collection,
                                                       timestamps)
    TimeSeriesSerializer().dump(sys.stdout, time_series, time_stamps)
