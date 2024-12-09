import os

import numpy as np
import pandas as pd
import tensorflow as tf

import models
import parsers
import utils


# VARS
batch_size = 64  # Batch size for training.
# epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
# num_samples = 10000  # Number of samples to train on.
# patience = 30

# Model savepoints
model_name = "hebrew"
style = "length"
version = f"3-{style}-same_model"
# version = "0-test"
ckpt_path = os.path.join("ckpt", model_name, version + ".h5")
log_path = os.path.join("log", model_name, version)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

# TODO-DONE drop EMES trop sefarim (iyov, mishlei, tehilim)
path = "../data/bhsac.tsv"
bhsac_df = pd.read_csv(path, sep="\t")
# Drop EMES sefarim
bhsac_df.drop(bhsac_df.loc[1041609:1219666].index, inplace=True)
print(bhsac_df["book"].unique())

lines = list(parsers.groupby_looper2(bhsac_df))

# Sort lines- this gives us pesukim from across different sefarim in our data printout later.
# Data is shuffled for training.
lines.sort(key=lambda x: len(x[0]), reverse=True)

# Turn the lines (one pasuk each) into character and text sets for the trop LSTM
input_characters, target_characters, input_texts, target_texts = parsers.compile_sets(lines=lines)

# For case of length-only binary model- Use simpler input texts for training.
original_input_texts = input_texts
if style == "length":
    # We only pass sentence length into the model
    input_characters, input_texts = parsers.grammar_to_length(original_input_texts)
elif style == "reverse":
    # We reverse the order of the trop (to test whether forcing starts from end of pasuk- should give better accuracy if so, right?)
    target_texts = [list(reversed(x)) for x in target_texts]

# Aggregate data into arrays
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
# Convert sets from compile_sets() into categorical data tables ready for the LSTM
# TODO-DECIDE save character dictionaries with model so we can use same tokenization for inference later? They SHOULD be index safe but very fishy
# encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index = parsers.build_lstm_data(input_characters=input_characters, target_characters=target_characters, input_texts=input_texts, target_texts=target_texts)
# Index safe
input_arr, target_inputs, target_labels, input_token_index, target_token_index = parsers.build_transformer_data(input_characters=input_characters, target_characters=target_characters, input_texts=input_texts, target_texts=target_texts)

# TODO-DONE build datasets
# ds = tf.data.Dataset.from_tensor_slices(((input_arr, target_inputs), target_labels)).batch(64)
idx = np.array(range(len(lines))).reshape(-1, 1)
ds = tf.data.Dataset.from_tensor_slices(((input_arr, target_inputs), target_labels, idx)).shuffle(buffer_size=20000, reshuffle_each_iteration=False).batch(64)
train_test_split = int(16000/64)
# idx is for inference later- we can use it to get the pasuk for output to the log file. We could also get node info with this.
ds_train_with_idx = ds.take(train_test_split)
ds_test_with_idx = ds.skip(train_test_split)
ds_train = ds_train_with_idx.map(lambda x, y, z: (x, y))
ds_test = ds_test_with_idx.map(lambda x, y, z: (x, y))

transformer = models.Transformer(num_layers=1, d_model=latent_dim, num_heads=1, dff=latent_dim, input_vocab_size=len(input_token_index)+1, target_vocab_size=len(target_token_index)+1)
learning_rate = models.CustomSchedule(d_model=latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(loss=models.masked_loss, optimizer=optimizer, metrics=[models.masked_accuracy])

early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, min_delta=0.005)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
tboard = tf.keras.callbacks.TensorBoard(log_dir=log_path)

# TRAIN
history = transformer.fit(ds_train, validation_data=ds_test, epochs=80, callbacks=[early_stopper, tboard])

# TODO word node per trop and per grammar token.
# models.tokenizer = Tokenizer
grammar_tokenizer = models.Tokenizer(dict=input_token_index)
trop_tokenizer = models.Tokenizer(dict=target_token_index)
accuracies = []

with open(os.path.join("log", f"{model_name}_{version}_validation_generated.txt"), "w", encoding='utf-8') as valid_file:
    for ((g_tokens, t_tokens), _, seq_index) in list(ds_test_with_idx.unbatch()):
        # g_tokens = grammar_tokenizer.tokenize(input_text)
        # t_tokens = trop_tokenizer.tokenize(target_text)

        stop = False
        output = [trop_tokenizer.tokenize(["START"])[0]]

        start, end = trop_tokenizer.tokenize([])

        while stop is False:
            predictions = transformer([g_tokens[tf.newaxis], np.array(output)[np.newaxis]], training=False)

            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.
            predicted_id = tf.argmax(predictions, axis=-1)

            if (predicted_id[0] == end) or (len(output) > 100):
                stop = True
            else:
                output.append(predicted_id.numpy()[0, 0])

        predicted_trop = trop_tokenizer.detokenize(output)
        accuracy = np.mean([x == y for (x, y) in zip(output, t_tokens)])
        accuracies.append(accuracy)
        print("-", file=valid_file)
        # print("Input verse:  ", lines[seq_index][0], file=valid_file)
        print("Input verse:  ", lines[seq_index.numpy()[0]][0], file=valid_file)
        print("Input grammar:", grammar_tokenizer.detokenize(g_tokens.numpy()), file=valid_file)
        print("Actual trop:  ", [utils.trop_names[x] for x in trop_tokenizer.detokenize(t_tokens.numpy())], file=valid_file)
        print("Decoded trop: ", [utils.trop_names[x] for x in trop_tokenizer.detokenize(output)], file=valid_file)
        print("Accuracy:     ", accuracy, file=valid_file)
    print(f"Overall accuracy (test): {np.mean(accuracies)}", file=valid_file)
transformer.save_weights(ckpt_path)
# transformer.load_weights(ckpt_path)

# TODO-DONE reverse trop model
# TODO-DONE save model
print("Done!")
