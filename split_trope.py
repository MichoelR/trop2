import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import utils


def split_trope(sentence):
    """For each word in a sentence, split into (word, trope)."""
    expr = "[" + "".join(utils.trops) + "]"
    sent_trope = []
    # TODO return character number as well so we know where trope was in word
    for word in sentence.split(" "):
        res = re.search(expr, word)
        # Only zero/one trope per word.
        try:
            word_trope = res.group(0)
        except AttributeError:
            continue

        sent_trope.append((word, word_trope))
    return sent_trope


# VARS
# Hebrew Engish parallel bible
print("Getting paths...")
paths = glob.glob("./hebrew_eng_parallel_bible/Hebrew/*/*#001.txt")

# MAIN
# Perakim
perek_df = pd.DataFrame(data={"paths": paths}).sort_values(by="paths")
perek_df["sefer"] = perek_df["paths"].str.split("/").apply(lambda x: x[3])
perek_df["perek"] = perek_df["paths"].str.split("/").apply(lambda x: x[4].split(".")[0]).str.extract("(\d+)").astype(int)
perek_df["perek"] = perek_df["perek"].apply(lambda x: x % 1000)
print("Reading tanakh...")
perek_df["all_pesukim"] = perek_df["paths"].apply(lambda x: open(x).read().replace("\u200e\xa0\u200f", "").split("\n"))

# Split perakim into pesukim
print("Splitting into pesukim...")
pasuk_df = perek_df["all_pesukim"].apply(pd.Series, 1).stack().to_frame()
pasuk_df.columns = ["pasuk"]
pasuk_df["verse"] = (pasuk_df.index.get_level_values(1) + 1).astype(int)
pasuk_df.index = pasuk_df.index.droplevel(-1)
torah_df = perek_df.join(pasuk_df)

# Sort pesukim by their order in tanakh (traditional)
ordered_sefarim = ['gen', 'exo', 'lev', 'num', 'deu', 'jos', 'jdg', 'sa1', 'sa2',\
                   'kg1', 'kg2', 'isa', 'jer', 'eze', 'hos', 'joe', 'amo', 'oba',\
                   'jon', 'mic', 'nah', 'hab', 'zep', 'hag', 'zac', 'mal', 'psa',\
                   'pro', 'job', 'sol', 'rut', 'lam', 'ecc', 'est', 'dan', 'ezr',\
                   'neh', 'ch1', 'ch2',]

torah_df["sefer_num"] = torah_df["sefer"].apply(ordered_sefarim.index)
torah_df = torah_df.sort_values(by=["sefer_num", "perek", "verse"])

# Final order for pesukim
torah_df = torah_df[["sefer", "perek",  "verse", "pasuk"]]

# Extract trope
print("Extracting trop...")
torah_df["word_trope"] = torah_df["pasuk"].apply(split_trope)
# Remove null lines
# torah_df["just_trop"] = torah_df.word_trope.apply(lambda x: x[0][1])
# torah_df["is_a_none"] = torah_df["just_trop"].isna()
# torah_df = torah_df[~torah_df["is_a_none"]]
# Reset useless index
torah_df = torah_df.reset_index(drop=True)

# TODO decision tree for (trope, next_trope) or (trope, previous_trope)
#  Test the following models:
#  trope, previous_trope
#  trope, next_trope
#  word, trope
#  and rank them by accuracy %

