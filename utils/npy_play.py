#!/usr/bin/python3
import numpy as np
import argparse
import cv2

def main():
	global args
	mean = np.load(args.mean)
	images = np.load(args.images)
	length = images.shape[2]
	for ind in range(length):
		if ind % args.ts:
			continue
		img = images[:,:,ind]+mean
		cv2.imshow('video', img)
		k = cv2.waitKey(500)
		if k == 27:
			return

if __name__ == "__main__":
	global args
	parser = argparse.ArgumentParser(description='Video player')
	parser.add_argument('-i', dest='images', required=True, type=str)
	parser.add_argument('-m', dest='mean', required=True, type=str)
	parser.add_argument('-s', dest='ts', required=True, type=int)
	args = parser.parse_args()

	try:
		main()

	except Exception as e:
		print(e)