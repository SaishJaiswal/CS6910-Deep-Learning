from LoadData import ReadData
import CNNmodel
import numpy as np
import keras
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import pdb
import config

from tensorflow.keras.models import load_model

def main(args):

	################################## Reading Arguments ##################################
	n_classes = args.n_classes
	n_filters = args.n_filters
	filter_size = args.filter_size
	filter_multiplier = args.filter_multiplier
	var_n_filters = args.var_n_filters
	l_rate = args.l_rate
	epochs = args.epochs
	optimizer = args.optimizer
	activation = args.activation
	loss = args.loss
	batch_size = args.batch_size
	initializer = args.initializer
	data_augmentation = args.data_augmentation
	denselayer_size = args.denselayer_size
	batch_norm = args.batch_norm
	train_model = args.train_model

	n_filters_layer1 = 32
	n_filters_layer2 = 32
	n_filters_layer3 = 32
	n_filters_layer4 = 32
	n_filters_layer5 = 32

	filter_shape = (filter_size, filter_size)

	DROP_OUT = 0.4

	############################################ Data Preprocessing ############################################
	WIDTH, HEIGHT, CHANNELS = 224, 224, 3
	train_data_dir = '/cbr/saish/Datasets/inaturalist_12K/train/'
	test_data_dir = '/cbr/saish/Datasets/inaturalist_12K/test/'

	################# Read Data #################
	X_train, X_val, X_test, y_train, y_val, y_test = ReadData(WIDTH, HEIGHT, CHANNELS, train_data_dir, test_data_dir, read_data=False)

	################# Data Augmentation #################
	if data_augmentation:
		train_datagen = ImageDataGenerator(
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='nearest')

	if train_model:
		model = CNNmodel.CNN_Model(n_classes, n_filters, filter_size, filter_multiplier, var_n_filters, l_rate, epochs, optimizer, activation, loss, batch_size, initializer, data_augmentation, denselayer_size, batch_norm, train_model)
		model.TrainModel(X_train, y_train, X_val, y_val)
		model.TestModel(X_test, y_test)

	else:	
		######################## Load the Model ########################
		model = load_model('model.h5')

		################################### Testing ###################################
		test_eval = model.evaluate(X_test, y_test, verbose=0)
		print('Test loss:', test_eval[0])
		print('Test accuracy:', test_eval[1])


############################ Main Funtion ############################
if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
