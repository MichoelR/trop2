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

import models
import parsers
import utils

# gpu memory config
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# VARS
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
patience = 30
model_name = "hebrew"
style = "full"
# version = f"3-{style}-patience={patience}"
version = "0-test"
ckpt_path = os.path.join("ckpt", model_name, version + ".h5")
log_path = os.path.join("log", model_name, version)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)


# DATA
# Vectorize the data.

# First, convert bhsac tree into lines of tuples.
# TODO drop EMES trop sefarim (iyov, mishlei, tehilim)
path = "../data/bhsac.tsv"
bhsac_df = pd.read_csv(path, sep="\t")
lines = list(parsers.groupby_looper2(bhsac_df))

# Sort lines
# TODO this reduces the complexity of the training set since we only use first N. Bad.
#  It also means that we can't even predict for longer pesukim, I think, since max sequence length is a hyperparam.
lines.sort(key=lambda x: len(x[0]))

# Turn the lines (one pasuk each) into character and text sets for the trop LSTM
input_characters, target_characters, input_texts, target_texts = parsers.compile_sets(lines=lines, num_samples=num_samples)

# For case of length-only binary model- Use simpler input texts for training.
original_input_texts = input_texts
if style == "length":
    # We only pass sentence length into the model
    input_characters, input_texts = parsers.grammar_to_length(original_input_texts)

# Aggregate data into arrays
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
# Convert sets from compile_sets() into categorical data tables ready for the LSTM
encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index = parsers.build_lstm_data(input_characters=input_characters, target_characters=target_characters, input_texts=input_texts, target_texts=target_texts)

# Create tf.data.datasets
# We need this crazy idx trick to figure out our train-test split for our own validation.
idx = range(num_samples)
ds = tf.data.Dataset.from_tensor_slices(((encoder_input_data, decoder_input_data), decoder_target_data, np.array(idx).reshape(-1,1)))
ds = ds.shuffle(buffer_size=num_samples, reshuffle_each_iteration=False).batch(batch_size)
split = int((num_samples * .9) / batch_size)
ds_train = ds.take(split)
ds_test = ds.skip(split)

# Pull our indexes for later experiments- note that we drop the last batch for simplicity.
ds_train_idx = ds_train.map(lambda x, y, z: z)
train_idx = np.array(list(ds_train_idx)[:-1]).reshape(-1)
ds_train = ds_train.map(lambda x, y, z: (x, y))

ds_test_idx = ds_test.map(lambda x, y, z: z)
test_idx = np.array(list(ds_test_idx)[:-1]).reshape(-1)
ds_test = ds_test.map(lambda x, y, z: (x, y))

train_idx.sort()
test_idx.sort()

# MODEL
model = models.lstm(num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens, latent_dim=latent_dim)
# Callbacks
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.005)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
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
# Restore the trained model and construct the encoder and decoder sub-models.
# We can also start by loading a model from here and run inference.
# TODO refactor inference into separate module.
model = keras.models.load_model(ckpt_path)
encoder_model, decoder_model = models.lstm_sampling_models(model)

# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

print(f"num test elems in train (should be zero!): {np.isin(test_idx, train_idx).sum()}")

# TODO store decoded pesukim in machine readable format. Add original pesukim (with original trop) to format.
with open(os.path.join("log", f"{model_name}_{version}_validation_generated.txt"), "w") as valid_file:
    for seq_index in test_idx:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        # decoded_sentence = decode_sequence(input_seq)
        decoded_sentence = models.decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index, target_token_index, max_decoder_seq_length)
        print("-", file=valid_file)
        # print("Input sentence:", input_texts[seq_index], file=valid_file)
        print("Input sentence:", original_input_texts[seq_index], file=valid_file)
        print("True output:", [utils.trop_names[x] for x in target_texts[seq_index][1:-1]], file=valid_file)
        print("Decoded sentence:", [utils.trop_names[x] for x in list(decoded_sentence[:-1])], file=valid_file)

print("Done!")
# TODO transformers experiment.
