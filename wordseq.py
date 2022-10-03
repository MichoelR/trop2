import pandas as pd
import seq2seq as s2
import parsers

df = pd.read_csv("data/bhsac.tsv", sep="\t")

# The verses will match up already since they each loop the same way
data, label = (parsers.split_all_grammar(df), parsers.split_all_trope(df))



model = s2.SimpleSeq2Seq()