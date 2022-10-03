"""Collection of parsers for machine learning."""
import re

import numpy as np
import pandas as pd

import utils


def groupby_looper2(bhsac_df):
    """Yields a single verse at a time, in order, with human readable identifier."""
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
                        if all_trope[-1] == chr(1433):
                            word_trope = all_trope[-1]
                        else:
                            word_trope = all_trope[0]
                        sent_trope.append(word_trope)
                        sent_trope_nodes.append(row.n)
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