import os

import numpy as np

import models
import utils

from tensorflow import keras


model_name = "hebrew"
version = "0-test"
ckpt_path = f"ckpt/{model_name}/{version}.h5"
model = keras.models.load_model(ckpt_path)
encoder_model, decoder_model = models.lstm_sampling_models(model)

# TODO easiest way to fix data issues is to generate once and store on disk (tf.dataset?)
# TODO input_ and target_token_index should be hardcoded and stored in a file. As is they are unreliable.
# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# TODO-DECIDE save and restore test_idx? is it index safe?
# TODO we won't need this test so ignore train_idx
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
