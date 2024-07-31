import pandas as pd
import numpy as np
import pickle
import gzip
import os

# Workaround to handle compatibility issues
from pandas.compat import pickle_compat as pkl

with gzip.open('CCNCS_Internship/Grad_CAM_output_data/heatmap.pickle', 'rb') as f:  
    # Using compatibility loader
    heatmap = pkl.load(f)

heatmap = heatmap.reset_index(drop=True)
heat_word = heatmap['kw'].values
heat_score = heatmap['heat'].values

mal_scoreCNT = 0
be_scoreCNT = 0
zero_scoreCNT = 0

for i in heat_score:
    if i > 0:
        be_scoreCNT += 1
    elif i < 0:
        mal_scoreCNT += 1 
    elif float(i) == float(0):
        zero_scoreCNT += 1

be_word = heat_word[0:be_scoreCNT]
be_word = be_word.tolist()

mal_heat_word = heat_word[(be_scoreCNT + zero_scoreCNT):len(heat_score)]
mal_heat_word = mal_heat_word.tolist()

print("mal :", len(mal_heat_word))
print("be :", len(be_word))
# print(heat_score)

with gzip.open('CCNCS_Internship/CNN/output_data/mal_feature_acg.pickle', 'wb') as f:
    pickle.dump(mal_heat_word, f)

print(mal_heat_word)

with gzip.open('CCNCS_Internship/CNN/output_data/be_feature_acg.pickle', 'wb') as f:
    pickle.dump(be_word, f)

print(be_word)

with gzip.open('CCNCS_Internship/CNN/output_data/output_acg.pickle', 'wb') as f:
    pickle.dump(mal_heat_word, f)
    pickle.dump(be_word , f)

