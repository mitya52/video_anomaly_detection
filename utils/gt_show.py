#!/usr/bin/python3

import numpy as np
import cv2
import sys
import glob

def main(uscd_dir_path):
	images = sorted(glob.iglob('{}/*.bmp'.format(uscd_dir_path)))
	for fn in images:
		im = cv2.imread(fn)
		cv2.imshow('video', im)
		k = cv2.waitKey(10)
		if k == 27:
			return

if __name__ == "__main__":
	try:
		main(sys.argv[1])

	except Exception as e:
		print(e)