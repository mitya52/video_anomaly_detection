#!/usr/bin/python3
import numpy as np
import os
import cv2
import argparse
import glob

from config import input_shape, temporal_saturation
from sklearn.mixture import GaussianMixture

def load(image_path):
	img = cv2.imread(image_path, 0)
	if args.vis:
		cv2.imshow('original grayscale', img)
		if cv2.waitKey(10) == 27:
			exit()
	w, h = input_shape[:2]
	img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
	return img / 256.

def load_video(video):
	_, video_fn = os.path.split(video)
	frames_dir = '{}/frames/{}'.format(args.source, video_fn)
	
	os.mkdir(frames_dir)
	os.system('ffmpeg -i {} -r {} {}/%04d.jpg'.format(video, args.fps*temporal_saturation, frames_dir))
	
	images = sorted(glob.iglob('{}/*.jpg'.format(frames_dir)))
	data = np.array([load(fn) for fn in images])
	mean = np.sum(data, axis=0) / data.shape[0]

	if args.vis:
		cv2.imshow('mean', mean)
		k = cv2.waitKey(1)

	np.save('{}/{}.npy'.format(args.source, video_fn), data)
	return mean

def preprocess_video(video, mean):
	_, video_fn = os.path.split(video)
	data = np.load('{}/{}.npy'.format(args.source, video_fn))

	bs, h, w = data.shape
	temp = np.zeros((h, w, bs))
	for i, d in enumerate(data):
		temp[:,:,i] = d - mean

	np.save('{}/{}.npy'.format(args.source, video_fn), temp)

def find_gmm(videos):
	data = np.array([])
	for video in videos:
		_, video_fn = os.path.split(video)
		x = np.load('{}/{}.npy'.format(args.source, video_fn)).ravel()
		np.random.shuffle(x)
		data = np.hstack((data, x[:min(x.shape[0], 10000)]))
	np.random.shuffle(data)
	data = data[:min(x.shape[0], 100000)]
	gmm = GaussianMixture(n_components=1)
	print('Compute gmm...')
	gmm.fit(data.reshape(-1, 1))
	m, s = gmm.means_.ravel()[0], gmm.covariances_.ravel()[0]
	print('Mean {}, sigma {}'.format(m, s))
	return np.array([m, s])

def main():
	videos = list(glob.iglob('{}/*.avi'.format(args.source)))
	print(videos)
	
	mean = []
	for video in videos:
		print('process {}...'.format(video))
		mean.append(load_video(video))

	mean = np.array(mean)
	mean = np.sum(mean, axis=0) / mean.shape[0]
	if args.vis:
		cv2.imshow('mean', mean)
		cv2.waitKey(1)

	for video in videos:
		print('process {}...'.format(video))
		preprocess_video(video, mean)
	np.save('{}/mean.npy'.format(args.source), mean)
	np.save('{}/gmm.npy'.format(args.source), find_gmm(videos))

if __name__ == "__main__":
	global args

	parser = argparse.ArgumentParser(description='Source Video path')
	parser.add_argument('-s', dest='source', required=True, type=str)
	parser.add_argument('-f', dest='fps', default=5, type=float)
	parser.add_argument('--vis', dest='vis', action='store_true')
	args = parser.parse_args()

	frames_dir = '{}/frames'.format(args.source)
	try:
		os.mkdir(frames_dir)
		main()

	except Exception as e:
		print(e)

	os.system('rm -rf {}'.format(frames_dir))