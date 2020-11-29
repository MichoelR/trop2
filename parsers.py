"""Collection of parsers for machine learning."""
import re

import numpy as np
import pandas as pd

import utils


# def groupby_looper(bhsac_df):
#     """Yields a single verse at a time, in order, with human readable identifier."""
#     # Groupby looping
#     gby = bhsac_df.groupby(by="in.verse")
#     for g in gby:
#         verse_id = bhsac_df[bhsac_df.n == g[0]][["book", "chapter", "verse", "n"]]
#         yield g[1], verse_id
#
# def split_all_grammar(bhsac_df):
#     """Returns a single grammar sequence for each verse."""
#     for verse, id in groupby_looper(bhsac_df):
#         seq = []
#         for i, row in verse.iterrows():
#             # labels
#             if row.otype == "clause_atom":
#                 seq.append(row.typ)
#             elif row.otype == "phrase_atom":
#                 seq.append(row.typ)
#             elif row.otype == "word":
#                 seq.append(row.pdp)
#             # flags
#             elif row.otype == "sentence_atom":
#                 seq.append("sentence_atom")
#             # else:
#             #     raise ValueError("unrecognized otype")
#         yield seq, id
#
# def split_trope(verse):
#     """For each word in a verse, split into (word, trope)."""
#     expr = "[" + "".join(utils.trops) + "]"
#     sent_trope = []
#     # TODO return character number as well so we know where trope was in word
#     for word in verse.split(" "):
#         res = re.search(expr, word)
#         # Only zero/one trope per word.
#         try:
#             word_trope = res.group(0)
#         except AttributeError:
#             continue
#
#         sent_trope.append((word, word_trope))
#
#     # # last word gets sof pasuk- won't get caught by regex because sof pasuk is not in g_word_utf8
#     # sent_trope.append((verse.split(" ")[-1], chr(1475)))
#     return sent_trope
#
# def split_all_trope(bhsac_df):
#     for verse, id in groupby_looper(bhsac_df):
#         verse_str = " ".join(verse.g_word_utf8.dropna()) + chr(1475)
#         yield split_trope(verse_str), id

# TODO single loop read
def groupby_looper2(bhsac_df):
    """Yields a single verse at a time, in order, with human readable identifier."""
    # Groupby looping
    pattern = "[" + "".join(utils.trops) + "]"
    gby = bhsac_df.groupby(by="in.verse")

    for g in gby:
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
                # qere-ketiv- have to check for nan and 'x is np.nan' doesn't work on this column so we do this nonsense.
                if isinstance(row.qere_utf8, str) and len(row.qere_utf8) > 0:
                    word = row.qere_utf8
                sent_word.append(word)
                sent_word_nodes.append(row.n)

                # trope
                try:
                    res = re.search(pattern, word)
                    word_trope = res.group(0)
                    sent_trope.append(word_trope)
                    sent_trope_nodes.append(row.n)
                # if no trope
                except AttributeError:
                    pass

        # Get that sof pasuk
        sent_word.append(verse.g_word_utf8.iloc[-1])
        sent_word_nodes.append(verse.n.iloc[-1])
        sent_trope.append(chr(1475))
        sent_trope_nodes.append(verse.n.iloc[-1])

        yield sent_word, sent_gram, sent_trope, verse_id


if __name__ == "__main__":
    path = "data/bhsac.tsv"
    bhsac_df = pd.read_csv(path, sep="\t")
    looper = groupby_looper2(bhsac_df=bhsac_df)
    # df = pd.DataFrame(data=looper)
    # x = next(looper)
    # data = zip(split_all_grammar(bhsac_df), split_all_trope(bhsac_df))
    # gram = split_all_grammar(bhsac_df)
    # trop = split_all_trope(bhsac_df)
    # df = pd.DataFrame(data=[gram, trop])
    # x = next(data)
    # y = next(data)
    # z = next(data)
    #list(zip(x[0][0], x[1][0]))
    # TODO save results
    with open("data/grammar_v_trop-v2.txt", "wb") as grammar_v_trop:
        for pasuk, grammar, trope, id in looper:
            grammar_v_trop.write("\t".join([str(pasuk), str(grammar), str(trope), str(id.values.tolist()[0])]).encode("utf-8"))
            grammar_v_trop.write("\n".encode("utf-8"))

    print("Done!")
# ((1,2), ((3,4),5)) -> [1,3,4,2,5]
