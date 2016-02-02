SHELL:=bash -i
DATADIR:=/home/vikasdhi/data
MIDDIR:=/home/vikasdhi/mid
PROJDATADIR:=$(DATADIR)/articulatedslam/2016-01-22
PROJMIDDIR:=$(MIDDIR)/articulatedslam/2016-01-22
PYTHONPATH:=simulation/:$(PYTHONPATH)

.SECONDARY:

bags:=all_static_2016-01-22-13-49-34 planar_2016-01-22-14-43-28 prism_2016-01-22-14-20-53 rev_2016-01-22-13-56-28 rev_2016-01-22-14-10-45 rev2_2016-01-22-14-32-13 rev_pris_2016-01-22-13-40-33 rev_prism_planar_2016-01-22-15-05-59
targets:=$(foreach b,$(bags),$(PROJMIDDIR)/$(b)/extracttrajectories_GFTT_SIFT.avi) $(foreach b,$(bags),$(PROJMIDDIR)/$(b)/extracttrajectories_GFTT_SIFT_odom_gt_timeseries.txt)#$(foreach b,$(bags),$(PROJMIDDIR)/$(b)/densetraj.gz) $(foreach b,$(bags),$(PROJMIDDIR)/$(b)/densetraj.avi)
all: $(targets) 

# Data dir to MID DIR
$(PROJMIDDIR)/%.bag: $(PROJDATADIR)/%.bag
	if [ -e $@ ] ; then true; else ln -sT $< $@; fi

# Data dir to MID DIR
$(PROJMIDDIR)/%/robot.txt: $(PROJDATADIR)/%/robot.txt
	if [ -e $@ ] ; then true; else ln -sT $(dir $<) $(patsubst %/,%,$(dir $@)); fi

%/extracttrajectories_GFTT_SIFT_odom_gt_timeseries.txt: %/extracttrajectories_GFTT_SIFT_timeseries.pickle %.bag %_optitrack/robot.txt scripts/extract_gt_odom.py
	source /opt/ros/indigo/setup.bash && \
		python scripts/extract_gt_odom.py $< $(word 2,$^) $(word 3,$^) > $@

%/extracttrajectories_GFTT_SIFT_timeseries.pickle: %/extracttrajectories_GFTT_SIFT.pickle scripts/track_collection_to_time_series.py
	source /opt/ros/indigo/setup.bash && \
	    python scripts/track_collection_to_time_series.py < $< > $@

# Recipe to convert bag to 2D SIFT trajectories
%/extracttrajectories_GFTT_SIFT.avi %/extracttrajectories_GFTT_SIFT.pickle: %.bag
	mkdir -p $(dir $@) && \
	    source /opt/ros/indigo/setup.bash && \
	    python scripts/extracttrajectories.py $< $*/extracttrajectories_%s_%s.avi $*/extracttrajectories_%s_%s.pickle

# video -> dense trajectories
%/densetraj.gz %/densetraj0000.png: %.bag build/densetraj/src/densetraj/build/devel/lib/dense-trajectories/DenseTrackStab
	mkdir -p $(dir $@) && \
	     build/densetraj/src/densetraj/build/devel/lib/dense-trajectories/DenseTrackStab $< -O $*/densetraj%04d.png > $*/densetraj.gz

# Common conversion recipe from frames to video
%.avi: %0000.png
	avconv -framerate 12 -i $(subst 0000.png,%04d.png,$<) -r 30 -vb 2M $@

# Download and make dense trajectories executable
# build/Makefile: CMakeLists.txt
# 	source /opt/ros/indigo/setup.bash && \
# 	    mkdir -p build && cd build && cmake ..
# 
# build/densetraj/src/densetraj/Makefile: build/Makefile
# 	source /opt/ros/indigo/setup.bash && \
# 	    make -C build
# 
# build/densetraj/src/densetraj/release/DenseTrackStab: build/densetraj/src/densetraj/Makefile
# 	make -C $(dir $<)
