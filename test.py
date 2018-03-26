#!/usr/bin/python3
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from keras.models import load_model
from config import temporal_depth, temporal_saturation, normalize_minmax

def mean_squared_loss(x1,x2):
	diff = (x1 - x2)
	sq_diff = diff**2
	#sq_diff = np.clip(sq_diff, 0, 0.02)
	mse = np.sqrt(sq_diff.sum()) / diff.shape[0]
	mask = normalize_minmax(sq_diff)
	return mse, mask

def prepare_bunch(frames):
	X = np.zeros((1, 227, 227, temporal_depth, 1))
	indices = np.array([i*temporal_saturation for i in range(temporal_depth)])
	for i, d in enumerate(indices):
		X[0,:,:,i,0] = frames[d].squeeze()
	return X

def abnormal_score(e):
	global emin, emax

	emin = min(emin, e)
	emax = max(emax, e)
	sa = (e - emin) / emax
	return 1 - sa

def main(video, model, mean, sigma):
	if args.vis:
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(0, plot_width)
		ax.set_ylim(0, 1)
		line, = ax.plot([], [])

	frames = []
	scores = []
	length = video.shape[2]
	for ind in range(length):
		frame = video[:,:,ind].copy()

		cv2.imshow('video', frame + mean)
		#frame = cv2.resize(frame, (227,227), interpolation=cv2.INTER_AREA)
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 256. - mean
		frame = np.clip(frame, -1, 1)
		#frame = normalize_minmax(frame, (-1, 1))

		if len(frames) == temporal_saturation*temporal_depth:
			frames = frames[1:] + [frame]
			
			X_test = prepare_bunch(frames)
			X_pred = model.predict(X_test)

			last_frame = lambda x: x[0,:,:,temporal_depth-1,0].squeeze()
			lf, lfp = last_frame(X_test), last_frame(X_pred)
			loss, mask = mean_squared_loss(lf, lfp)
			scores.append(abnormal_score(loss))
			
			#cv2.imshow('orig', lf)
			#cv2.imshow('pred', lfp)
			cv2.imshow('mask', mask)

		else:
			# accumulate frames
			frames.append(frame)

		if len(scores) > plot_width:
			scores = scores[1:]

		if len(scores) and args.vis:
			line.set_xdata(range(len(scores)))
			line.set_ydata(scores)
			plt.pause(0.001)

		k = cv2.waitKey(1)
		if k == 27:
			break


if __name__ == "__main__":
	global args
	global plot_width
	global emin, emax

	parser = argparse.ArgumentParser(description='Test model')
	parser.add_argument('--video', dest='video', required=True, type=str)
	parser.add_argument('--model', dest='model', required=True, type=str)
	parser.add_argument('--norm', dest='normalize', required=True, type=str)
	parser.add_argument('--vis', dest='vis', action='store_true')
	args = parser.parse_args()

	#cap = cv2.VideoCapture(args.video)
	video = np.load(args.video)
	model = load_model(args.model)
	mean = np.load('{}/mean.npy'.format(args.normalize))
	_, sigma = np.load('{}/gmm.npy'.format(args.normalize))

	plot_width = 300
	emin = 1e9
	emax = -1

	main(video, model, mean, sigma)
