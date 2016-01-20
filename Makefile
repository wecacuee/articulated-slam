SHELL:= bash -i
DATADIR:=/home/vikasdhi/data
MIDDIR:=/home/vikasdhi/mid
PROJDATADIR:=$(DATADIR)/articulatedslam/2016-01-15
PROJMIDDIR:=$(MIDDIR)/articulatedslam/2016-01-15

bags:=all_dynamic_2016-01-15-16-07-26 2016-01-15-16-54-57 2016-01-15-17-02-57 2016-01-15-17-11-53 2016-01-15-17-12-48 all_dynamic_2016-01-15-15-51-33 all_static_2016-01-15-15-39-50 only_rev_prism_2016-01-15-16-27-26 rev_and_chair_2016-01-15-17-03-58
targets:=$(foreach b,$(bags),$(PROJMIDDIR)/$(b)/densetraj.gz) $(foreach b,$(bags),$(PROJMIDDIR)/$(b)/densetraj.avi)
all: $(targets)
outputtraj:=$(PROJMIDDIR)/2016-01-15-17-12-48_extracttraj.pickle
densetrajout:=$(PROJMIDDIR)/2016-01-15-17-12-48/densetraj.gz

# Data dir to MID DIR
$(PROJMIDDIR)/%.bag: $(PROJDATADIR)/%.bag
	if [ -e $@ ] ; then true; else ln -sT $< $@; fi

# Recipe to convert bag to 2D SIFT trajectories
%/extracttrajectories0000.png %_extracttraj.pickle: %.bag simulation/extracttrajectories.py
	mkdir -p $(dir $@) && \
	    source /opt/ros/indigo/setup.bash && \
	    python simulation/extracttrajectories.py $< $(subst 0000.png,%04d.png,$@)


# bag -> video
%/inputbagframe0000.png: %.bag launch/bag2video.launch
	mkdir -p $(dir $@) && \
	    source /opt/ros/indigo/setup.bash && \
	    roslaunch launch/bag2video.launch bag:=$< out:=$(subst 0000.png,%04d.png,$@)

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
