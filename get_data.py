#!/usr/bin/python
from subprocess import call
import os
import sys

if __name__=='__main__':
    # Get the directory of the script
    dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    # Make the data directory
    call(["mkdir",os.path.join(dir_path,'ros_recorded')])
    # Get the data
    call(["wget","https://doc-10-3g-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/dqq1c7oguvq58g2udnb8rkuhvulgneko/1428285600000/02602031496084045619/*/0B4fmrLclwph7MlBYSElucnBxVWc?e=download","-O",os.path.join(dir_path,'ros_recorded','data_recorded_door.bag')])

