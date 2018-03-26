import numpy as np

input_shape = (227, 227, 1) # grayscale
batch_size = 16
temporal_depth = 10
temporal_saturation = 3 # netx frame in bunch
skipping_strides = [0.7, 0.8, 0.9, 1.0] # less than 1.0!

def normalize_minmax(x, ival=(0, 1)):
	a, b = ival
	return (b - a)*(x - np.min(x))/(np.max(x) - np.min(x)) + a