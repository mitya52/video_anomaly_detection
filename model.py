from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose,BatchNormalization
from keras.models import Sequential

import os

def add_layers(model):
	model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
	model.add(BatchNormalization())
	model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))

	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
	model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))

	model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
	model.add(BatchNormalization())
	model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

def add_layers_half(model):
	model.add(Conv3D(filters=64,kernel_size=(11,11,1),strides=(4,4,1),kernel_initializer='lecun_uniform',padding='valid',input_shape=(227,227,10,1),activation='tanh'))
	model.add(BatchNormalization())
	model.add(Conv3D(filters=32,kernel_size=(5,5,1),strides=(2,2,1),kernel_initializer='lecun_uniform',padding='valid',activation='tanh'))

	model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,kernel_initializer='random_uniform',padding='same',dropout=0.4,return_sequences=True,recurrent_dropout=0.3))
	model.add(ConvLSTM2D(filters=16,kernel_size=(3,3),strides=1,kernel_initializer='random_uniform',padding='same',dropout=0.4,return_sequences=True))
	model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,kernel_initializer='random_uniform',padding='same',dropout=0.4,return_sequences=True))

	model.add(Conv3DTranspose(filters=64,kernel_size=(5,5,1),strides=(2,2,1),kernel_initializer='random_uniform',padding='valid',activation='tanh'))
	model.add(BatchNormalization())
	model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),kernel_initializer='random_uniform',padding='valid',activation='tanh'))

def load_model(model_fn='model.h5'):
	model=Sequential()
	add_layers(model)

	if os.path.exists(model_fn):
		model.load_weights(model_fn)	

	return model