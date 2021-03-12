import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import FeedForwardNN
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator


def main(args):
	print(args)
	if args.optimizer not in ['sgd', 'mgd', 'nag', 'rmsprop', 'adam', 'nadam']:
		exit("Not a valid optimizer!")

	# Hyperparameters
	layer_sizes = args.layer_sizes
	L = len(layer_sizes)
	epochs = args.epochs
	l_rate = args.l_rate
	optimizer = args.optimizer
	activation_func = args.activation
	loss_func = args.loss
	output_activation = args.output_activation
	batch_size = args.batch_size


	# Load dataset           
	(X_train, Y_train), (x_test, y_test) = fashion_mnist.load_data()

	X_train = X_train.astype('float64')
	Y_train = Y_train.astype('float64')
	x_test = x_test.astype('float64')
	y_test = y_test.astype('float64')

	scaler = StandardScaler()

	X_train = X_train.reshape(len(X_train),784)
	#X_train = (X_train/255).astype('float32')	# Normalize the images
	X_train = scaler.fit_transform(X_train)
	Y_train = Y_train.reshape(len(Y_train),1)
	Y_train = to_categorical(Y_train)

	x_test = x_test.reshape(len(x_test), 784)
	#x_test = (x_test/255).astype('float32')
	x_test = scaler.fit_transform(x_test)
	y_test = y_test.reshape(len(y_test), 1)
	y_test = to_categorical(y_test)

	# Split the training dataset into train and validation sets
	x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

	# Creating an object of the class FFNN
	network = FeedForwardNN.FFNN(layer_sizes, L, epochs, l_rate, optimizer, batch_size, activation_func, loss_func, output_activation)
	network.train(x_train, y_train, x_val, y_val)
	test_acc, test_loss = network.modelPerformance(x_test, y_test)
	print("################################")
	print("Testing Accuracy = " + str(test_acc))
	print("Testing Loss = " + str(test_loss))


if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
