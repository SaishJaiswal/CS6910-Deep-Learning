import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, Nadam
from keras.layers import CuDNNGRU as LSTM
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Embedding, RepeatVector, Dropout 
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from numpy.random import shuffle
from tqdm import tqdm
import csv
import math
import pdb

# In order to run Wandb
WANDB = 0

if WANDB:
	import wandb
	from wandb.keras import WandbCallback
	wandb.init(config={"batch_size": 64, "epochs": 10, "Cell_Type": "LSTM", "hidden_layer_size": 64, "dropout": 0.2}, project="Deep-Learning-RNN")
	myconfig = wandb.config


epochs = 5              # Number of epochs to train for.
l_rate = 0.01
hidden_layer_size = 128                 # Latent dimensionality of the encoding space.
emb_size  =  64
batch_size = 64                 # Batch size for training.
dropout = 0
n_enc_layers = 2
Cell_Type = 'GRU'
beam_size = 5


Languages = {'Bengali': 'bn', 'Gujarati': 'gu', 'Hindi': 'hi', 'Kannada': 'kn', 'Malayalam': 'ml', 'Marathi': 'mr', 'Punjabi': 'pa', 'Sindhi': 'sd', 'Sinhala': 'si', 'Tamil': 'ta', 'Telugu': 'te', 'Urdu': 'ur'}
Language = Languages['Hindi']

############################## Path to the dataset ##############################
train_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.train.tsv'
val_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.dev.tsv'
test_data_path = '/cbr/saish/Datasets/dakshina_dataset_v1.0/'+ Language +'/lexicons/'+ Language +'.translit.sampled.test.tsv'

############################## Reading Data ##############################
def ReadData(data_path, category, inp_chars=None, tar_chars=None, max_enc_seq_len=None, max_dec_seq_len=None, num_enc_tok=None, num_dec_tok=None, inp_tok_index=None, tar_tok_index=None):
	input_texts = []
	target_texts = []
	input_characters = set()
	target_characters = set()

	with open(data_path) as tsv_file:
	
		tsvreader = csv.reader(tsv_file, delimiter="\t")
		for line in tsvreader:
			input_text = line[1:2]
			target_text = line[0:1]

			input_text = ' '.join([str(elem) for elem in input_text])
			target_text = ' '.join([str(elem) for elem in target_text])

			target_text = "#" + target_text + "$"

			input_texts.append(input_text)
			target_texts.append(target_text)
		
			for char in input_text:
				if char not in input_characters:
					input_characters.add(char)
			for char in target_text:
				if char not in target_characters:
					target_characters.add(char)	

	############ Appending Spaces ############
	input_texts.append(" ")
	target_texts.append(" ")
	input_characters.add(" ")
	target_characters.add(" ")

	if category == "train":
		input_characters = sorted(list(input_characters))
		target_characters = sorted(list(target_characters))
		num_encoder_tokens = len(input_characters)
		num_decoder_tokens = len(target_characters)
		max_encoder_seq_length = max([len(txt) for txt in input_texts])
		max_decoder_seq_length = max([len(txt) for txt in target_texts])
	elif category == "val" or category == "test":
		input_characters = inp_chars
		target_characters = tar_chars
		num_encoder_tokens = num_enc_tok
		num_decoder_tokens = num_dec_tok
		max_encoder_seq_length = max_enc_seq_len
		max_decoder_seq_length = max_dec_seq_len
		
		
	print("Number of samples:", len(input_texts))
	print("Number of unique input tokens:", num_encoder_tokens)
	print("Number of unique output tokens:", num_decoder_tokens)
	print("Max sequence length for inputs:", max_encoder_seq_length)
	print("Max sequence length for outputs:", max_decoder_seq_length)

	input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
	target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

	encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
	decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
	decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

	for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
	
		for t, char in enumerate(input_text):
			encoder_input_data[i, t] = input_token_index[char]

		encoder_input_data[i, t + 1 :] = input_token_index[" "]

		for t, char in enumerate(target_text):
			decoder_input_data[i, t] = target_token_index[char]
			if t > 0:
				decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

		decoder_input_data[i, t + 1 :] = target_token_index[" "]
		decoder_target_data[i, t:, target_token_index[" "]] = 1.0

	tsv_file. close()

	return input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data

############################## Reading Train Data ##############################
input_texts, target_texts, input_characters, target_characters, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data, decoder_input_data, decoder_target_data = ReadData(train_data_path, "train")

############################## Reading Validation Data ##############################
val_input_texts, val_target_texts, _, _, _, _, _, _, _, _, val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = ReadData(val_data_path, "val", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

############################## Reading Test Data ##############################
test_input_texts, test_target_texts, _, _, _, _, _, _, _, _, test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = ReadData(test_data_path, "test", input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index)

################################ Build A Model ################################

encoder_inputs = keras.Input(shape=(None,))
enc_emb =  layers.Embedding(num_encoder_tokens, emb_size)(encoder_inputs)



if Cell_Type == 'RNN':	
	encoder_rnn = tf.keras.layers.SimpleRNN(hidden_layer_size, return_state=True, return_sequences=True, name="encoder1")
	encoder_outputs, h1 = encoder_rnn(enc_emb)
	_, h2 = tf.keras.layers.SimpleRNN(hidden_layer_size, return_state=True, name="encoder2")(encoder_outputs)
	encoder_states = [h1, h2]
elif Cell_Type == 'LSTM':
	encoder_lstm = tf.keras.layers.LSTM(hidden_layer_size, return_state=True, return_sequences=True, name="encoder1") 
	e_outputs, h1, c1 = encoder_lstm(enc_emb)
	_, h2, c2 = tf.keras.layers.LSTM(hidden_layer_size, return_state=True, name="encoder2")(e_outputs)
	encoder_states = [h1, c1, h2, c2]
	
elif Cell_Type == 'GRU':
	encoder_gru = tf.keras.layers.CuDNNGRU(hidden_layer_size, return_state=True,return_sequences=True, name="encoder1")
	encoder_outputs, h1 = encoder_gru(enc_emb)
	_, h2 = tf.keras.layers.CuDNNGRU(hidden_layer_size, return_state=True, name="encoder2")(encoder_outputs)
	encoder_states = [h1, h2]

decoder_inputs = keras.Input(shape=(None,))
dec_emb_layer = layers.Embedding(num_decoder_tokens, hidden_layer_size)
dec_emb = dec_emb_layer(decoder_inputs)

if Cell_Type == 'RNN':
	decoder_rnn_layer1 = keras.layers.SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="decoder1")
	d_outputs, dh1 = decoder_rnn_layer1(dec_emb, initial_state=[h1])
	decoder_rnn_layer2 = keras.layers.SimpleRNN(hidden_layer_size, return_sequences=True, return_state=True, name="decoder2")
	final, dh2 = decoder_rnn_layer2(d_outputs, initial_state= [h2])
elif Cell_Type == 'LSTM':
	decoder_lstm_layer1 = tf.keras.layers.LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="decoder1")
	d_outputs, dh1, dc1 = decoder_lstm_layer1(dec_emb, initial_state= [h1, c1])
	decoder_lstm_layer2 = tf.keras.layers.LSTM(hidden_layer_size, return_sequences=True, return_state=True, name="decoder2")
	final, dh2, dc2 = decoder_lstm_layer2(d_outputs, initial_state= [h2, c2])
	
elif Cell_Type == 'GRU':
	decoder_gru_layer1 = keras.layers.CuDNNGRU(hidden_layer_size, return_sequences=True, return_state=True, name="decoder1")
	d_outputs, dh1 = decoder_gru_layer1(dec_emb, initial_state= [h1])
	decoder_gru_layer2 = tf.keras.layers.CuDNNGRU(hidden_layer_size, return_sequences=True, return_state=True, name="decoder2")
	final, dh2 = decoder_gru_layer2(d_outputs, initial_state= [h2])

decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(final)


model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

################################ Train the Model ################################

opt = Adam(l_rate, beta_1=0.9, beta_2=0.999, decay=0.001)
#opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)

model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=[categorical_accuracy])
model.summary()
#pdb.set_trace()


if WANDB:

	model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True, 
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
    callbacks=[wandb.keras.WandbCallback()]
)
else:

	model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
)

#wandb.log({"loss": loss})

################################ Save the Model ################################
#model.save("s2s")

################### Restore the model and construct the encoder and decoder ###################

#model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]

if Cell_Type == 'RNN':
	
	encoder_states = [h1, h2]
	encoder_model = keras.Model(encoder_inputs, encoder_states)
	decoder_inputs = model.input[1]
	
	decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
	decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
	
	dec_emb_layer = model.layers[3]
	dec_emb2 = dec_emb_layer(decoder_inputs)

	decoder_rnn_layer1 = model.layers[5]
	decoder_rnn_layer2 = model.layers[7]
	d_out, state_h = decoder_rnn_layer1(dec_emb2, initial_state=decoder_states_inputs[:1])
	d_out, state_h1 = decoder_rnn_layer2(d_out, initial_state=decoder_states_inputs[-1:])
	decoder_states = [state_h, state_h1]

elif Cell_Type == 'LSTM':
	
	encoder_states = [h1, c1, h2, c2]
	encoder_model = keras.Model(encoder_inputs, encoder_states)
	#pdb.set_trace()
	decoder_inputs = model.input[1]
	
	decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
	decoder_state_input_c = keras.Input(shape=(hidden_layer_size,))
	decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
	decoder_state_input_c1 = keras.Input(shape=(hidden_layer_size,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_h1, decoder_state_input_c1]

	dec_emb_layer = model.layers[3]         
	dec_emb2 = dec_emb_layer(decoder_inputs)
	decoder_lstm_layer1 = model.layers[5]
	decoder_lstm_layer2 = model.layers[7]
	d_out, state_h, state_c = decoder_lstm_layer1(dec_emb2, initial_state=decoder_states_inputs[:2])
	d_out, state_h1, state_c1 = decoder_lstm_layer2(d_out, initial_state=decoder_states_inputs[-2:])
	decoder_states = [state_h, state_c, state_h1, state_c1]
	
	
elif Cell_Type == 'GRU':

	encoder_states = [h1, h2]
	encoder_model = keras.Model(encoder_inputs, encoder_states)

	decoder_inputs = model.input[1]
	decoder_state_input_h = keras.Input(shape=(hidden_layer_size,))
	decoder_state_input_h1 = keras.Input(shape=(hidden_layer_size,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_h1]
	
	dec_emb_layer = model.layers[3]
	dec_emb2 = dec_emb_layer(decoder_inputs)

	decoder_gru_layer1 = model.layers[5]
	decoder_gru_layer2 = model.layers[7]
	d_out, state_h = decoder_gru_layer1(dec_emb2, initial_state=decoder_states_inputs[:1])
	d_out, state_h1 = decoder_gru_layer2(d_out, initial_state=decoder_states_inputs[-1:])
	decoder_states = [state_h, state_h1]

decoder_dense = model.layers[8]
d_out = decoder_dense(d_out)
# Final decoder model
decoder_model = keras.Model(
	[decoder_inputs] + decoder_states_inputs,
	[d_out] + decoder_states)

decoder_model.summary()


# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def beam_search_decoder(predictions, top_k):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]
   # print("predictions: ", predictions)
    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []
      #  print("Token_probs: ", token_probs)
        #pdb.set_trace()
        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
               
               # print(abs(token_probs[char_index]))		
                #considering log-likelihood for scoring
                likelihood = token_probs[char_index]
                
                new_score = old_score + math.log(likelihood)
                new_sequences.append((new_seq, new_score))
                
        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
        #select top-k based on score 
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]
        #print("Output_sequences: ", output_sequences)
        
    return output_sequences


def beam_decode_sequence(input_seq, beam_size):
	# Encode the input as state vectors.
	if Cell_Type == 'RNN':
		states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'LSTM':
		states_value = encoder_model.predict(input_seq)
	elif Cell_Type == 'GRU':
		states_value = encoder_model.predict(input_seq)
	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1,1))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = target_token_index['#']

	decoded_sentence = ''
	stop_condition = False
	output_prob = []
	
	while not stop_condition:
		if Cell_Type == 'RNN':
			output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
		elif Cell_Type == 'LSTM':
			output_tokens, h, c, h1, c1 = decoder_model.predict([target_seq] + states_value)			
		elif Cell_Type == 'GRU':
			output_tokens, h, h1 = decoder_model.predict([target_seq] + states_value)
			
		
		output_prob.append(output_tokens.tolist())         		
		
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length or find stop character.
		if sampled_char == "$" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True
			output_prob = sum(output_prob, [])
			output = [elem for twod in output_prob for elem in twod]
			#pdb.set_trace()
			sequences = beam_search_decoder(output, beam_size)	
		else:
			target_seq = np.zeros((1, 1))
			target_seq[0, 0] = sampled_token_index

			# Update states
			if Cell_Type == 'RNN':
				states_value = [h, h1]
			elif Cell_Type == 'LSTM':
				states_value = [h, c, h1, c1]
			elif Cell_Type == 'GRU':
				states_value = [h, h1]

	return sequences

def decode_the_word(decoded_sequence, beam_size):	
	
	suggestions = list()
	for k in range(beam_size):
		decoded_word = ''		
		encoded_list = decoded_sequence[k][0]
		for i in range(len(encoded_list)):
			sampled_token_index = decoded_sequence[k][0][i]
			sampled_char = reverse_target_char_index[sampled_token_index]
			decoded_word += sampled_char					
	
		suggestions.append(decoded_word[:-1])	
		
	return suggestions
	
val_words = len(val_input_texts)
train_words = len(input_texts)
test_words = len(test_input_texts)

print('\n CALCULATING WORD-LEVEL ACCURACY!! \n')

def calculateAccuracy(encoder_input_data, target_texts, total_words):

	count = 0
	for seq_index in tqdm(range(total_words), desc='Transliteration in Progress'):
		
		input_seq = encoder_input_data[seq_index : seq_index + 1]
		decoded_sequence = beam_decode_sequence(input_seq, beam_size)
		suggestions = decode_the_word(decoded_sequence, beam_size)
		
		if target_texts[seq_index][1:-1] in suggestions:
			count = count + 1
	print("Count: ", count)	
	accuracy = count/total_words
	return accuracy
	
acc = calculateAccuracy(val_encoder_input_data, val_target_texts, val_words)
print("Testing Accuracy (word-wise): %f " % (acc))

