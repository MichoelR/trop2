"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2020/04/26
Description: Character-level recurrent sequence-to-sequence model.
"""
"""
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

"""
## Setup
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import parsers
import utils

# gpu memory config
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# VARS
model_name = "hebrew"
version = "1-short_pesukim"
ckpt_path = os.path.join("ckpt", model_name, version + ".h5")
log_path = os.path.join("log", model_name, version)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# DATA
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

path = "../data/bhsac.tsv"
bhsac_df = pd.read_csv(path, sep="\t")
lines = list(parsers.groupby_looper2(bhsac_df))

# Sort lines
lines.sort(key=lambda x: len(x[0]))

for line in lines[: min(num_samples, len(lines) - 1)]:
    pasuk, input_text, target_text, _, _, _, _ = line
    input_text = input_text.copy()
    target_text = target_text.copy()

    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text.insert(0, "\t")
    target_text.append("\n")
    input_texts.append(input_text)
    target_texts.append(target_text)

    # Compile sets
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# Aggregate
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

# Index all input and output chars
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Compile numerical datasets
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    # encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    # decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    # decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# Create dataset
# We need this crazy idx trick to figure out our train-test split for our own validation.
idx = range(num_samples)
ds = tf.data.Dataset.from_tensor_slices(((encoder_input_data, decoder_input_data), decoder_target_data, np.array(idx).reshape(-1,1)))
ds = ds.shuffle(buffer_size=num_samples).batch(batch_size)
split = int((num_samples * .9) / batch_size)
ds_train = ds.take(split)
ds_test = ds.skip(split)

# Pull of indexes for later experiments- note that we drop the last batch for simplicity.
ds_train_idx = ds_train.map(lambda x, y, z: z)
train_idx = np.array(list(ds_train_idx)[:-1]).reshape(-1)
ds_train = ds_train.map(lambda x, y, z: (x, y))

ds_test_idx = ds_test.map(lambda x, y, z: z)
test_idx = np.array(list(ds_test_idx)[:-1]).reshape(-1)
ds_test = ds_test.map(lambda x, y, z: (x, y))

# MODEL
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Callbacks
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.005)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='loss')
tboard = tf.keras.callbacks.TensorBoard(log_dir=log_path)

# TRAIN
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
    callbacks=[early_stopper, model_checkpoint, tboard],
)
# Save model
model.save(ckpt_path)

"""
## Run inference (sampling)

1. encode input and retrieve initial decoder state
2. run one step of decoder with this initial state
and a "start of sequence" token as target.
Output will be the next target token.
3. Repeat with the current target token and current states
"""

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model(ckpt_path)

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


"""
You can now generate decoded sentences as such:
"""

with open("validation_generated.txt", "w") as valid_file:
    for seq_index in test_idx:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print("-", file=valid_file)
        print("Input sentence:", input_texts[seq_index], file=valid_file)
        print("True output:", [utils.trop_names[x] for x in target_texts[seq_index][1:-1]], file=valid_file)
        print("Decoded sentence:", [utils.trop_names[x] for x in list(decoded_sentence[:-1])], file=valid_file)

print("Done!")