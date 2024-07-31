import pandas as pd
import numpy as np
import pickle
import gzip
import os

folder = 'CCNCS_Internship/CNN/data/'
target = os.listdir(folder)

with gzip.open('./output_data/be_feature_acg.pickle', 'rb') as f:
    be_feature_acg = pickle.load(f)
with gzip.open('./output_data/mal_feature_acg.pickle', 'rb') as f:
    feature_acg = pickle.load(f)
with gzip.open('CCNCS_Internship/CNN/output_data/output.pickle', 'rb') as f:
    data = pickle.load(f)

def removeAllOccur(a, i):
    return [x for x in a if x != i]

X_name = data[0].values
X = data.drop(columns=[0, 'class']).values.flatten().tolist()  # Convert to a 1D list
X = removeAllOccur(removeAllOccur(X, str(0)), 0)

be_target = os.path.join(folder, "be.csv")
mal_target = os.path.join(folder, "mal.csv")
be_score_be = np.zeros(len(be_target), dtype=np.float32)
be_score = np.zeros(len(be_target), dtype=np.float32)

for i in range(len(be_target)):
    tokenized_doc1 = be_feature_acg
    tokenized_doc2 = X[i]
    tokenized_doc4 = feature_acg

    intersection_be = set(tokenized_doc1).intersection(set(tokenized_doc2))
    intersection = set(tokenized_doc4).intersection(set(tokenized_doc2))

    be_score_be[i] = len(intersection_be) / len(tokenized_doc1)
    be_score[i] = len(intersection) / len(tokenized_doc4)

mal_score = np.zeros(len(mal_target), dtype=np.float32)
mal_score_be = np.zeros(len(mal_target), dtype=np.float32)

for i in range(len(mal_target)):
    print("here")
    tokenized_doc1 = be_feature_acg
    tokenized_doc2 = X[len(be_target) + i]
    tokenized_doc4 = feature_acg

    intersection_be = set(tokenized_doc1).intersection(set(tokenized_doc2))
    intersection = set(tokenized_doc4).intersection(set(tokenized_doc2))

    mal_score_be[i] = len(intersection_be) / len(tokenized_doc1)
    mal_score[i] = len(intersection) / len(tokenized_doc4)

print(f"Length of be_score: {len(be_score)}")
print(f"Length of mal_score: {len(mal_score)}")
print(f"Length of be_score_be: {len(be_score_be)}")
print(f"Length of mal_score_be: {len(mal_score_be)}")

print(mal_score_be)

result = pd.DataFrame({'be_score': be_score, 'mal_score': mal_score})
result1 = pd.DataFrame({'be_score_be': be_score_be, 'mal_score_be': mal_score_be})

print("be aver_be", np.mean(be_score_be))
print("mal aver_be", np.mean(mal_score_be))
print("be aver", np.mean(be_score))
print("mal aver", np.mean(mal_score))

with open('./output_data/jacaard_score.pickle', 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

with open('./output_data/be_jacaard_score.pickle', 'wb') as f:
    pickle.dump(result1, f, pickle.HIGHEST_PROTOCOL)

# with open('./output_data/jacaard_score_be.pickle', 'wb') as f:
#     pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
# with open('./output_data/jacaard_score_mal.pickle', 'wb') as f:
#     pickle.dump(result_mal, f, pickle.HIGHEST_PROTOCOL)
# with open('./output_data/be_jacaard_score_be.pickle', 'wb') as f:
#     pickle.dump(result_be_score_be, f, pickle.HIGHEST_PROTOCOL)
# with open('./output_data/be_jacaard_score_mal.pickle', 'wb') as f:
#     pickle.dump(result_mal_score_be, f, pickle.HIGHEST_PROTOCOL)




