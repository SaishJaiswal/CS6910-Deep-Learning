import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU, CuDNNLSTM, CuDNNGRU
from tqdm import tqdm

from DecodeText import DecodeSequence
from Accuracy import CalculateAccuracy

# In order to run Wandb
WANDB = 0

if WANDB:
	import wandb
	from wandb.keras import WandbCallback
	wandb.init(config={"batch_size": 64, "epochs": 10, "Cell_Type": "LSTM", "h_layer_size": 64, "emb_size": 64, "dropout": 0}, project="Deep-Learning-RNN")
	myconfig = wandb.config

class RNN_Model():
	
	##################### Initialize the Hyperparameters #####################
	def __init__(self):


	##################### Defining Model Architecture #####################
	def InitializeModel(self):


	##################### Train Model #####################
	def TrainModel(self, X_train, y_train, X_val, y_val):


	##################### Test Model #####################
	def TestModel(self, X_test, y_test):
