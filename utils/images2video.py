#!/usr/bin/python3

import numpy as np
import glob
import cv2
import sys

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (227, 227))

files = sorted(glob.iglob('{}/*.tif'.format(sys.argv[1])))
for fn in files:
	frame = cv2.imread(fn)
	frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
	out.write(frame)
	cv2.imshow('frame', frame)
	cv2.waitKey(10)

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
