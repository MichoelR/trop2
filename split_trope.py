import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import re
import os


def split_trope(sentence):
    """For each word in a sentence, split into (word, trope)."""
    all_trope = "".join([chr(x) for x in range(1425, 1455)] + [chr(1469)])
    expr = "[" + all_trope + "]"
    sent_trope = []
    # TODO return character number as well so we know where trope was in word
    for word in sentence.split(" "):
        res = re.search(expr, word)
        # print("word:", word)
        # print("res:", res)
        # print("res.0", res.group(0))
        # Only zero/one trope per word.
        try:
            word_trope = res.group(0)
        except AttributeError:
            word_trope = None

        sent_trope.append((word, word_trope))
    return sent_trope


# VARS
# Hebrew Engish parallel bible
paths = glob.glob("./hebrew_eng_parallel_bible/Hebrew/*/*#001.txt")

# MAIN
# Perakim
perek_df = pd.DataFrame(data={"paths": paths}).sort_values(by="paths")
perek_df["sefer"] = perek_df["paths"].str.split("/").apply(lambda x: x[3])
perek_df["perek"] = perek_df["paths"].str.split("/").apply(lambda x: x[4].split(".")[0]).str.extract("(\d+)").astype(int)
perek_df["perek"] = perek_df["perek"].apply(lambda x: x % 1000)
perek_df["all_pesukim"] = perek_df["paths"].apply(lambda x: open(x).read().replace("\u200e\xa0\u200f", "").split("\n"))

# Split perakim into pesukim
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
torah_df["word_trope"] = torah_df["pasuk"].apply(split_trope)

#
# def get_random_pasuk():
#     sefer = random.choice(sefarim)
# ##    pasuk =
    
# ch = open(paths[0]).readlines()
# sent = open(paths[0]).readline()
# print("sent:", sent)
# print(split_trope(sent))

print("Done!")