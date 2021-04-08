import numpy as np
import keras
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K 
from keras.layers import Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU

#import PIL
import tensorflow as tf


############################################ Data Preprocessing ############################################
img_width, img_height = 224, 224

train_data_dir = '/cbr/saish/Datasets/inaturalist_12K/train/'
validation_data_dir = '/cbr/saish/Datasets/inaturalist_12K/val/'
nb_train_samples = 1000
nb_validation_samples = 200

'''
inaturalist_images_dict = {
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

X, y = [], []

for species_name, images in inaturalist_images_dict.items():
	print("##################### " + species_name + " #####################")
	for image in tqdm(images, total=len(images)):
		img = cv2.imread(str(image))
		resized_img = cv2.resize(img,(img_width, img_height))
		X.append(resized_img)
		y.append(inaturalist_labels_dict[species_name])

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=0)


X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 3)
X_val = X_val.reshape(X_val.shape[0], img_width, img_height, 3)
y_train = y_train.reshape(len(y_train), 1)
y_val = y_val.reshape(len(y_val), 1)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

np.savez('data.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
'''

'''
############################################## Data Augmentation ##############################################
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

generator = train_datagen.flow_from_directory(
    '/cbr/saish/Datasets/inaturalist_12K/train/',
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia'],
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir='/cbr/saish/Datasets/inatAugmented',
    save_prefix=None,
    save_format="jpg",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)

print("##################################### Generating augmented images #####################################")
i=0
for batch in generator:
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely


pdb.set_trace()
'''

#### Read the Stored Data
data = np.load('/cbr/saish/Datasets/data.npz')
X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255.
X_val = X_val / 255.

################################### Defining Model Architecture ###################################
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def gelu(x):
	"""Gaussian Error Linear Unit.
	This is a smoother version of the RELU.
	Original paper: https://arxiv.org/abs/1606.08415
	refer : https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
	Args:
	x: float Tensor to perform activation.
	Returns:
	`x` with the GELU activation applied.
	"""
	cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    	return x * cdf

pdb.set_trace()

model = Sequential()
initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)

#Variance scaling initializer 
#keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

#Orthogonal Initializer
#keras.initializers.Orthogonal(gain=1.0, seed=None)

#he_normal Initializer
#keras.initializers.he_normal(seed=None)

'''
############################## Layer 1 ##############################
model.add(Conv2D(128, (2, 2), activation='linear', kernel_initializer=initializer, padding='same', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

############################## Layer 2 ##############################
model.add(Conv2D(64, (2, 2), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

############################## Layer 3 ##############################
model.add(Conv2D(64, (2, 2), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

############################## Layer 4 ##############################
model.add(Conv2D(32, (2, 2), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))

############################## Layer 5 ##############################
model.add(Conv2D(16, (2, 2), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
'''
############################## Layer 1 ##############################
model.add(Conv2D(128, (3, 3), activation='linear', kernel_initializer=initializer, padding='same', input_shape=input_shape))
model.add(Activation('gelu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############################## Layer 1 ##############################
model.add(Conv2D(128, (3, 3), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(Activation('gelu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############################## Layer 1 ##############################
model.add(Conv2D(128, (3, 3), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(Activation('gelu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############################## Layer 1 ##############################
model.add(Conv2D(128, (3, 3), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(Activation('gelu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############################## Layer 1 ##############################
model.add(Conv2D(128, (3, 3), activation='linear', kernel_initializer=initializer, padding='same'))
model.add(Activation('gelu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############################## Dense Layer 1 ##############################
model.add(Flatten())
model.add(Dense(128, activation='linear', kernel_initializer=initializer))
model.add(Activation('gelu'))

############################## Dense Layer 2 ##############################
#model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

################################### Training ###################################
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val), verbose=1)


################################### Testing ###################################
test_eval = model.evaluate(X_val, y_val, verbose=1)   
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

print("Accuracy = " + str(test_eval))
    
#pdb.set_trace()
