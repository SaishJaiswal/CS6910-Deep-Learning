import config
import FeedForwardNN
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def display_images(images, titles, cols, cmap):

	rows = len(images)//cols + 1
	plt.figure(figsize=(12,6))
	i = 1
	for image, title in zip(images, titles):
		plt.subplot(rows, cols, i)
		plt.title(title, fontsize=12)
		plt.axis('off')
		plt.imshow(image, cmap=cmap)
		i += 1

	plt.show()
	plt.savefig('/cbr/saish/PhD/image.png')


def main(args):
	print(args)

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
	initializer = args.initializer

	# Load dataset           
	(X_train, Y_train), (x_test, y_test) = fashion_mnist.load_data()

	# Display images
	indexes = [0,1,3,5,6,8,16,18,19,23]
	images = [X_train[i] for i in indexes]
	titles = ['Ankle Boot', 'T-Shirt', 'Dress', 'Pullover', 'Sneaker', 'Sandal', 'Trouser', 'Shirt', 'Coat', 'Bag']
	display_images(images, titles, cols=5, cmap=plt.get_cmap('gray'))
	
	# Change data type to float64
	X_train = X_train.astype('float64')
	Y_train = Y_train.astype('float64')
	x_test = x_test.astype('float64')
	y_test = y_test.astype('float64')

	# Normalize the images - mean-variance
	scaler = StandardScaler()

	X_train = X_train.reshape(len(X_train),784)
	#X_train = (X_train/255).astype('float32')
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
	network = FeedForwardNN.FFNN(layer_sizes, L, epochs, l_rate, optimizer, batch_size, activation_func, loss_func, output_activation, initializer)

	# Training the network
	network.train(x_train, y_train, x_val, y_val)

	# Testing
	test_acc, test_loss = network.modelPerformance(x_test, y_test)
	print("################################")
	print("Testing Accuracy = " + str(test_acc))
	print("Testing Loss = " + str(test_loss))


if __name__ == "__main__":
	args = config.parseArguments()
	main(args)
