"""Collection of parsers for machine learning."""
import re

import numpy as np
import pandas as pd

import utils


def groupby_looper2(bhsac_df):
    """Yields a single verse at a time, in order, with human readable identifier.

    For trop: meteg is not included, sof pasuk is, we only take one trop per word."""
    # Groupby looping
    pattern = "[" + "".join(utils.trops) + "]"
    gby = bhsac_df.groupby(by="in.verse")

    for verse_count, g in enumerate(gby):
        if not (verse_count % 1000): print(verse_count)
        # n = starting point of verse
        verse_id = bhsac_df[bhsac_df.n == g[0]][["book", "chapter", "verse", "n"]]
        verse = g[1]

        sent_word = []
        sent_gram = []
        sent_trope = []

        sent_word_nodes = []
        sent_gram_nodes = []
        sent_trope_nodes = []

        for i, row in verse.iterrows():
            # Grammar
            # labels
            if row.otype == "clause":
                sent_gram.append(row.kind)
                sent_gram_nodes.append(row.n)
            elif row.otype == "phrase_atom":
                sent_gram.append(row.typ)
                sent_gram_nodes.append(row.n)
            elif row.otype == "word":
                sent_gram.append(row.pdp)
                sent_gram_nodes.append(row.n)
            # flags
            elif row.otype == "sentence_atom":
                sent_gram.append("sentence_atom")
                sent_gram_nodes.append(row.n)

            # Words and trope
            if row.g_word_utf8 is not np.nan:
                # word
                word = row.g_word_utf8
                # print(word)
                # qere-ketiv- have to check for nan and 'x is np.nan' doesn't work on this column so we do this nonsense.
                if isinstance(row.qere_utf8, str) and len(row.qere_utf8) > 0:
                    word = row.qere_utf8
                sent_word.append(word)
                sent_word_nodes.append(row.n)

                # trope
                try:
                # res = re.search(pattern, word)
                    all_trope = re.findall(pattern, word)
                    if len(all_trope):
                        # chr(1433)=pashta- used to indicate the accented syllable. But doesn't this just show up as two pashtas? So why not take the first one?
                        # TODO-DONE check with ta whether this is the only double trop we need to worry about.
                        # TODO-DONE extract both trope in cases were the same trope is not doubled.
                        word_trope = set(all_trope)
                        # if all_trope[-1] == chr(1433):
                        #     word_trope = all_trope[-1]
                        # else:
                        #     word_trope = all_trope[0]
                        # sent_trope.append(word_trope)
                        sent_trope.extend(word_trope)
                        sent_trope_nodes.extend([row.n for x in range(len(all_trope))])
                    else:
                        pass
                        # print(f"skipping {word}")
                # if no trope
                except BaseException as e:
                    print(e)

        # Get that sof pasuk
        # sent_word.append(verse.g_word_utf8.iloc[-1])
        # sent_word_nodes.append(verse.n.iloc[-1])
        sent_trope.append(chr(1475))
        sent_trope_nodes.append(verse.n.iloc[-1])

        yield sent_word, sent_gram, sent_trope, sent_word_nodes, sent_gram_nodes, sent_trope_nodes, verse_id


def compile_sets(lines, num_samples=None):
    """Input comes from list(groupby_looper2(bhsac_df)). Turns the lines into character and text sets for the trop LSTM.
    The grammar will be in a prefix graph notation (a sentence with phrase label prepended to each phrase/clause).

    The data will still need further preprocessing (build_lstm_inputs())."""
    # Set of characters in each language
    input_characters = []
    target_characters = []
    # Set of texts for each language
    input_texts = []
    target_texts = []
    if not num_samples:
        num_samples = len(lines)

    for line in lines[:num_samples]:
        pasuk, input_text, target_text, _, _, _, _ = line
        input_text = input_text.copy()
        target_text = target_text.copy()

        # TODO if we use LSTM again, we should encode START and STOP characters in build_lstm_data() since we removed them here.
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        # target_text.insert(0, "\t")
        # target_text.append("\n")
        input_texts.append(input_text)
        target_texts.append(target_text)

        # Compile character sets
        for char in input_text:
            if char not in input_characters:
                input_characters.append(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.append(char)
    return input_characters, target_characters, input_texts, target_texts


def grammar_to_length(original_input_texts):
    """Turns a set of grammar texts into a set of binary sequences. This encodes only the length of the sentence.txt

    This allows us to test a null hypothesis- trop is dependent on the length of the pasuk. Alternatively, the
    model is only learning the length of the pasuk."""
    # You will need to save input_characters since our new data only encodes one character, not the full grammar alphabet.
    input_characters = ["x"]
    input_texts = []

    for sent in original_input_texts:
        simple_sent = []
        for char in sent:
            simple_sent.append("x")
        input_texts.append(simple_sent)

    return input_characters, input_texts


def build_lstm_data(input_characters, target_characters, input_texts, target_texts):
    """Converts sets from compile_sets() into categorical data tables ready for the LSTM."""
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
        # TODO-DECIDE do we need an input stop token? Why?
        # if style == "full": # skip for "length" only model, since rest will be zeroes.
        #     encoder_input_data[i, t+1:, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        # decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index


def build_transformer_data(input_characters, target_characters, input_texts, target_texts):
    # Index all input and output chars
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters, start=3)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters, start=3)])
    input_token_index["START"] = 1
    target_token_index["START"] = 1
    input_token_index["END"] = 2
    target_token_index["END"] = 2

    input_arr = np.zeros(shape=(len(input_texts), max([len(x) for x in input_texts])+2))
    target_inputs = np.zeros(shape=(len(input_texts), max([len(x) for x in target_texts])+2))
    target_labels = np.zeros(shape=(len(input_texts), max([len(x) for x in target_texts])+2))
    input_arr[:, 0] = input_token_index["START"]
    target_inputs[:, 0] = target_token_index["START"]

    for (i, text) in enumerate(input_texts):
        for (j, char) in enumerate(text):
            input_arr[i, j+1] = input_token_index[char]
        input_arr[i, j+2] = input_token_index["END"]

    for (i, text) in enumerate(target_texts):
        for (j, char) in enumerate(text):
            target_inputs[i, j+1] = target_token_index[char]
            target_labels[i, j] = target_token_index[char]
        target_labels[i, j+1] = target_token_index["END"]

    return input_arr, target_inputs, target_labels, input_token_index, target_token_index


if __name__ == "__main__":
    path = "data/bhsac.tsv"
    bhsac_df = pd.read_csv(path, sep="\t")
    looper = groupby_looper2(bhsac_df=bhsac_df)

    # TODO save results
    with open("data/grammar_v_trop-v2.2.txt", "wb") as grammar_v_trop:
        for sent_word, sent_gram, sent_trope, sent_word_nodes, sent_gram_nodes, sent_trope_nodes, verse_id in looper:
            grammar_v_trop.write("\t".join([str(sent_word), str(sent_gram), str(sent_trope), str(sent_word_nodes), str(sent_gram_nodes), str(sent_trope_nodes), str(verse_id.values.tolist()[0])]).encode("utf-8"))
            grammar_v_trop.write("\n".encode("utf-8"))

    print("Done!")
# ((1,2), ((3,4),5)) -> [1,3,4,2,5]
# Current version: v2.1
