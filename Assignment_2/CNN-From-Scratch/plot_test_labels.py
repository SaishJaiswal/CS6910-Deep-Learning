import numpy as np
import keras
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam


'''
* Number of filters: 32, 64
* Multiplier: 1, 0.5, 2
* Size of filters: 
* Data Augmentation: True, False
* Activation funtion in each layer: relu, leakyrelu
* Number of neurons in dense layer: 128, 64
* Batch Normalization: True, False
'''


def main(args):

	################################## Reading Arguments ##################################
	no_layers = 5
	var_n_filters = []
	n_classes = args.n_classes
	n_filters = args.n_filters
	filter_size = args.filter_size
	filter_multiplier = args.filter_multiplier
	#var_n_filters = args.var_n_filters
	for i in range(no_layers):
		var_n_filters.append(n_filters)
		n_filters = n_filters * filter_multiplier
	#var_n_filters.append(10)
	l_rate = args.l_rate
	epochs = args.epochs
	optimizer = args.optimizer
	activation = args.activation
	loss = args.loss
	batch_size = args.batch_size
	#initializer = args.initializer
	data_augmentation = args.data_augmentation
	denselayer_size = args.denselayer_size
	dropout = args.dropout
	batch_norm = args.batch_norm
	train_model = args.train_model


	filter_shape = (filter_size, filter_size)

	#DROP_OUT = 0.5

	n_filters_layer1 = 16
	n_filters_layer2 = 32
	n_filters_layer3 = 32
	n_filters_layer4 = 64
	n_filters_layer5 = 128


	class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

	############################################ Data Preprocessing ############################################
	WIDTH, HEIGHT, CHANNELS = 224, 224, 3
	train_data_dir = '/cbr/saish/Datasets/inaturalist_12K/train/'
	test_data_dir = '/cbr/saish/Datasets/inaturalist_12K/test/'

	################# Reading Data #################
	read_data = False

	if read_data:

		inaturalist_labels_dict = {
		    'Amphibia': 0,
		    'Animalia': 1,
		    'Arachnida': 2,
		    'Aves': 3,
		    'Fungi': 4,
		    'Insecta': 5,
		    'Mammalia': 6,
		    'Mollusca': 7,
		    'Plantae': 8,
		    'Reptilia': 9,
		}
	
		######### Read Train Data #########
		inaturalist_train_dict = {
		    'Amphibia': list(glob.glob(train_data_dir + 'Amphibia/*')),
		    'Animalia': list(glob.glob(train_data_dir + 'Animalia/*')),
		    'Arachnida': list(glob.glob(train_data_dir + 'Arachnida/*')),
		    'Aves': list(glob.glob(train_data_dir + 'Aves/*')),
		    'Fungi': list(glob.glob(train_data_dir + 'Fungi/*')),
		    'Insecta': list(glob.glob(train_data_dir + 'Insecta/*')),
		    'Mammalia': list(glob.glob(train_data_dir + 'Mammalia/*')),
		    'Mollusca': list(glob.glob(train_data_dir + 'Mollusca/*')),
		    'Plantae': list(glob.glob(train_data_dir + 'Plantae/*')),
		    'Reptilia': list(glob.glob(train_data_dir + 'Reptilia/*')),
		}

		X, y = [], []

		for species_name, images in inaturalist_train_dict.items():
			print("##################### " + species_name + " #####################")
			for image in tqdm(images, total=len(images)):
				img = cv2.imread(str(image))
				resized_img = cv2.resize(img,(WIDTH, HEIGHT))
				X.append(resized_img)
				y.append(inaturalist_labels_dict[species_name])
            
		######### Convert to Numpy Array #########
		X = np.array(X)
		y = np.array(y)

		######### Train-Test Split ##############
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=1, stratify=y)
		print(Counter(y_train))
		print(Counter(y_val))

		######### Reshape #########
		X_train = X_train.reshape(X_train.shape[0], WIDTH, HEIGHT, 3)
		X_val = X_val.reshape(X_val.shape[0], WIDTH, HEIGHT, 3)
		y_train = y_train.reshape(len(y_train), 1)
		y_val = y_val.reshape(len(y_val), 1)

		######### Convert to one-hot vector #########
		y_train = to_categorical(y_train)
		y_val = to_categorical(y_val)


		######### Read Test Data #########
		inaturalist_test_dict = {
		    'Amphibia': list(glob.glob(test_data_dir + 'Amphibia/*')),
		    'Animalia': list(glob.glob(test_data_dir + 'Animalia/*')),
		    'Arachnida': list(glob.glob(test_data_dir + 'Arachnida/*')),
		    'Aves': list(glob.glob(test_data_dir + 'Aves/*')),
		    'Fungi': list(glob.glob(test_data_dir + 'Fungi/*')),
		    'Insecta': list(glob.glob(test_data_dir + 'Insecta/*')),
		    'Mammalia': list(glob.glob(test_data_dir + 'Mammalia/*')),
		    'Mollusca': list(glob.glob(test_data_dir + 'Mollusca/*')),
		    'Plantae': list(glob.glob(test_data_dir + 'Plantae/*')),
		    'Reptilia': list(glob.glob(test_data_dir + 'Reptilia/*')),
		}

		X_test, y_test = [], []

		for species_name, images in inaturalist_test_dict.items():
			print("##################### " + species_name + " #####################")
			for image in tqdm(images, total=len(images)):
				img = cv2.imread(str(image))
				resized_img = cv2.resize(img,(WIDTH, HEIGHT))
				X_test.append(resized_img)
				y_test.append(inaturalist_labels_dict[species_name])

			

		######### Convert to Numpy Array #########
		X_test = np.array(X_test)
		y_test = np.array(y_test)

		######### Reshape #########
		X_test = X_test.reshape(X_test.shape[0], WIDTH, HEIGHT, 3)
		y_test = y_test.reshape(len(y_test), 1)

		######### Convert to one-hot vector #########
		y_test = to_categorical(y_test)
		
		np.savez('data.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, X_test=X_test, y_test=y_test)

	###### If the data is already saved #####
	else:

		################# Reading Stored Data #################
		data = np.load('/cbr/saish/Datasets/data.npz')
		X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['y_train'], data['y_val'], data['y_test']

		X_train = X_train.astype('float32')
		X_val = X_val.astype('float32')
		X_test = X_test.astype('float32')
		X_train = X_train / 255.
		X_val = X_val / 255.		
		X_test = X_test / 255.


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

	train_model = False

	if train_model:
		################################### Defining Model Architecture ###################################
		if K.image_data_format() == 'channels_first':
		    input_shape = (CHANNELS, WIDTH, HEIGHT)
		else:
		    input_shape = (WIDTH, HEIGHT, CHANNELS)


		############ Initialize Model ############
		model = Sequential()

		initializer = keras.initializers.Orthogonal(gain=1.0, seed=42)

		model.add(Conv2D(n_filters_layer1, filter_shape, activation='linear', kernel_initializer=initializer, padding='same', input_shape=input_shape))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(n_filters_layer2, filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(n_filters_layer3, filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(n_filters_layer4, filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(n_filters_layer5, filter_shape, activation='linear', kernel_initializer=initializer, padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(BatchNormalization())
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Flatten())
		model.add(Dense(denselayer_size, activation='linear', kernel_initializer=initializer))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dropout(dropout))

		model.add(Dense(n_classes, activation='softmax'))

		opt = Adam(l_rate)

		model.compile(opt, loss=loss, metrics=['accuracy'])

		model.summary()

		###################### Training ######################
		hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
		
			

		######################## Save the Model ########################
		model.save('model.h5')
		print("Model saved!!")

	else:	
		######################## Load the Model ########################
		model = load_model('model.h5')

	test_labels = []
	p = 1
	columns = 3
	classes = 10
	plt.figure(figsize=(30, 20))
	# make a prediction
	y_pred = model.predict_classes(X_test)
	# show the inputs and predicted outputs
	for i in range(len(X_test)):
		print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
		test_labels.append(y_pred[i])
		#plt.subplot(classes, 3, i+1)
		ax = plt.subplot(classes, 3, p)
		ax.set_xticks([])
		ax.set_yticks([])
		#plt.title('True label:' + class_names[y_test[i]], fontsize=30)
		plt.title('Predicted label:' + class_names[test_labels[i]], fontsize=25)
		plt.axis('off')
		plt.tight_layout()
		imgplot = plt.imshow(X_test[i])
		p = p+1
		if(p % 31==0):
			break
	plt.show()
	plt.savefig('test_images.jpg')


############################ Main Funtion ############################
if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
