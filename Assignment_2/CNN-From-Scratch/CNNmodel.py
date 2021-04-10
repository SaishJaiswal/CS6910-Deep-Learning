import numpy as np
import keras
import cv2
import glob
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam

WIDTH, HEIGHT, CHANNELS = 224, 224, 3
DROP_OUT = 0.4

class CNN_Model():
	# Initialize the Hyperparameters
	def __init__(self, n_classes, n_filters, filter_size, filter_multiplier, var_n_filters, l_rate, epochs, optimizer, activation, loss, batch_size, initializer, data_augmentation, denselayer_size, batch_norm, train_model):

		self.n_classes = n_classes
		self.n_filters = n_filters
		self.filter_size = filter_size
		self.filter_multiplier = filter_multiplier
		self.var_n_filters = var_n_filters
		self.l_rate = l_rate
		self.epochs = epochs
		self.optimizer = optimizer
		self.activation = activation
		self.loss = loss
		self.batch_size = batch_size
		self.initializer = initializer
		self.data_augmentation = data_augmentation
		self.denselayer_size = denselayer_size
		self.batch_norm = batch_norm
		self.train_model = train_model

		self.n_filters_layer1 = 32
		self.n_filters_layer2 = 32
		self.n_filters_layer3 = 32
		self.n_filters_layer4 = 32
		self.n_filters_layer5 = 32


		self.filter_shape = (self.filter_size, self.filter_size)

		self.model = self.InitializeModel()

		
	def InitializeModel(self):
		
		################################### Defining Model Architecture ###################################
		if K.image_data_format() == 'channels_first':
		    input_shape = (CHANNELS, WIDTH, HEIGHT)
		else:
		    input_shape = (WIDTH, HEIGHT, CHANNELS)


		############ Initialize Model ############
		model = Sequential()

		initializer = keras.initializers.Orthogonal(gain=1.0, seed=42)

		model.add(Conv2D(self.n_filters_layer1, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same', input_shape=input_shape))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(self.n_filters_layer2, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(self.n_filters_layer3, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(self.n_filters_layer4, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(self.n_filters_layer5, self.filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Flatten())
		model.add(Dense(self.denselayer_size, activation='linear', kernel_initializer=initializer))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(DROP_OUT))

		model.add(Dense(self.n_classes, activation='softmax'))

		opt = Adam(lr=self.l_rate)

		model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])

		model.summary()

		return model


	def TrainModel(self, X_train, y_train, X_val, y_val):

		##################### Train Model #####################
		history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1)
		return history

	def TestModel(self, X_test, y_test):
		
		################################### Testing ###################################
		test_eval = self.model.evaluate(X_test, y_test, verbose=0)
		print('Test loss:', test_eval[0])
		print('Test accuracy:', test_eval[1])
