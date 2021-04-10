from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, DenseNet201, MobileNet, ResNet50, InceptionV3, InceptionResNetV2, Xception
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras import backend as K
import pdb


HEIGHT, WIDTH, CHANNEL = 224, 224, 3
EPOCHS = 5
BATCH_SIZE=64

############################################ Data Preprocessing ############################################
img_width, img_height = 224, 224

train_data_dir = '/cbr/saish/Datasets/inaturalist_12K/train/'
validation_data_dir = '/cbr/saish/Datasets/inaturalist_12K/val/'
nb_train_samples = 1000
nb_validation_samples = 200

#################### Read the Stored Data ####################
data = np.load('/cbr/saish/Datasets/data.npz')
X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255.
X_val = X_val / 255.

###################### Data Augmentation ######################
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")


################################### Defining Model Architecture ###################################
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#pdb.set_trace()


######### Add Model Here ##############
baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(10, activation="softmax")(headModel)

from keras.utils.vis_utils import plot_model
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False


print(len(model.layers))
print(model.layers)
print(model.summary())

INIT_LR = 1e-4

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

#### Compile the model ####
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[['accuracy']])

print("Training the model...")
H = model.fit(trainAug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=(X_val, y_val), epochs=5)

print("Evaluating network...")
predIdxs = model.predict(X_val, batch_size=64)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(y_val.argmax(axis=1), predIdxs,target_names=lb.classes_))

count = 0;
for index in range(len(predIdxs)):  
	if (y_val.argmax(axis = 1)[index] == predIdxs[index]): 
		count += 1

Accuracy = count/len(predIdxs)
print("Testing Accuracy: " )
print(Accuracy)
print('=======================')
