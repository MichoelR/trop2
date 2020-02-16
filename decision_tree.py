import subprocess

import numpy as np
import pandas as pd

import utils

from scipy.spatial.distance import cosine
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder

from split_trope import torah_df


# Data
# We need trop + BEGIN and END labels
all_trope = sorted(np.array(utils.trops + ["BEGIN", "END"]))
trop_names = ["BEGIN", "END"] + [utils.trop_names[x] for x in utils.trops]

# Match each trope to the preceding trope.
print("Creating current-next labeled trope pairs...")
trop_pairs = []
for vrs in torah_df.word_trope:
    trop0 = "BEGIN"
    for wrd in vrs:
        trop = wrd[1]
        trop_pairs.append([trop0, trop])
        trop0 = trop
    trop_pairs.append([trop, "END"])

trop_pairs = np.array(trop_pairs)

# We will use first_trop to predict second_trop
first_trops = trop_pairs[:, 0].reshape(-1, 1)
second_trops = trop_pairs[:, 1].reshape(-1, 1)

# Encode trop as onehot embedding
# TODO-DONE link with all_trope in split_trope
enc = OneHotEncoder(categories=[all_trope], sparse=False)
enc.fit(np.append(first_trops, [["END"]], axis=0))

# Transform data and labels
first_ohe = enc.transform(first_trops)
second_ohe = enc.transform(second_trops)

# Train
dtc = DecisionTreeClassifier()
dtc.fit(first_ohe, second_ohe)

# TODO can we get labeled classes?
# Save visualization
export_graphviz(dtc, out_file="out/trop_tree.dot",
    # class_names=dtc.classes_,
    feature_names=trop_names)

# subprocess.call("dot -Tpng out/trop_tree.dot -o out/trop_tree.png")
# Predict
# TODO validation data
preds = dtc.predict(first_ohe)

# Evaluate model
error = mean_absolute_error(second_ohe, preds)
total_error = (second_ohe - preds).sum()
cos = cosine(second_ohe.flatten(), preds.flatten())
acc = accuracy_score(second_ohe, preds)
# bal_acc = balanced_accuracy_score(second_ohe, preds)
print("mean absolute error:", error)
print("total error:", total_error)
# Cosine distance seems like our best metric
print("cosine distance:", cos)
print("accuracy:", acc)
# print("balanced accuracy:", bal_acc)

#working
trop_data_df = pd.DataFrame(data=first_ohe, columns=trop_names)
trop_lbl_df = pd.DataFrame(data=second_ohe, columns=trop_names)
trop_pred_df = pd.DataFrame(data=preds, columns=trop_names)

print("done")
