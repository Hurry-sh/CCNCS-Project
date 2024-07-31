import os
import pickle
import gzip
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Setting the folder paths correctly
folder = 'CCNCS_Internship\\CNN\\data\\mal.csv'
be_folder = 'CCNCS_Internship\\CNN\\data\\be.csv'

# Check if the folder path is a directory or file
if os.path.isdir(folder):
    target = os.listdir(folder)
else:
    target = [folder]

if os.path.isdir(be_folder):
    be_target = os.listdir(be_folder)
else:
    be_target = [be_folder]

with gzip.open('CCNCS_Internship/CNN/output_data/train_x.pickle', 'rb') as f:
    train_x = pickle.load(f)

with gzip.open('CCNCS_Internship/CNN/output_data/keyword_rev_dict.pickle', 'rb') as f:
    keyword_rev_dict = pickle.load(f)

with gzip.open('CCNCS_Internship/CNN/output_data/api_call_word_list.pickle', 'rb') as f:
    all_values = pickle.load(f)

model = load_model('CNN_output_data/CNN.keras')
model.summary()

def grad_cam_conv1D(model, layer_name, x):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    x = np.expand_dims(x, axis=0)  # Ensure x is in the correct shape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        if isinstance(predictions, list):
            predictions = predictions[0]
        loss = predictions[0]  # Extract the loss from predictions
    grads = tape.gradient(loss, conv_outputs)[0]
    casted_conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    casted_grads = tf.cast(grads, tf.float32)
    weights = tf.reduce_mean(casted_grads, axis=0)
    grad_cam = np.zeros(casted_conv_outputs.shape[0], dtype=np.float32)
    for i, w in enumerate(weights):
        grad_cam += w * casted_conv_outputs[:, i]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam /= np.max(grad_cam)
    return grad_cam, np.expand_dims(grad_cam, axis=0)

all_values = all_values.iloc[0:, 0].values if isinstance(all_values, pd.DataFrame) else np.array(all_values)
val = [[] for _ in range(len(all_values))]

print("Grad CAM Extracting.....")

for idx in range(len(be_target), len(be_target) + len(target)):
    hm, graded = grad_cam_conv1D(model, 'conv1d', x=train_x[idx])
    kww = [keyword_rev_dict[i] for i in train_x[idx]]
    for j in range(len(kww)):
        for k in range(len(all_values)):
            if all_values[k] == kww[j]:
                try:
                    val[k].append(hm[j])
                except ValueError:
                    pass

last_val = [0]
print("Grad CAM Extracted")

print("Grad CAM Total.........")

for l in range(1, len(all_values)):
    temp = 0
    for m in range(len(val[l])):
        temp += val[l][m]
    temp /= (len(val[l]) if len(val[l]) > 0 else 1)  # Avoid division by zero
    last_val.append(temp)

hm_tbl = pd.DataFrame({'heat': last_val, 'kw': all_values})
hm_tbl = hm_tbl.sort_values(by=['heat'], axis=0, ascending=False)

output_folder = 'Grad_CAM_output_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with gzip.open(os.path.join(output_folder, 'heatmap.pickle'), 'wb') as f:
    pickle.dump(hm_tbl, f)

print(hm_tbl)
print("done")
