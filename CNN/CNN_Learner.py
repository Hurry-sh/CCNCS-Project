# import os
# import pandas as pd
# import numpy as np
# import pickle
# import gzip
# import tensorflow as tf

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import StratifiedKFold

# os.environ["CUDA_VISIBLE_DEVICES"] = ''
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e) 

# with gzip.open("CCNCS_Internship/CNN/output_data/train_x.pickle" , 'rb') as f:
#     train_x = pickle.load(f)

# with gzip.open("CCNCS_Internship/CNN/output_data/train_y.pickle" , 'rb') as f:
#     train_y = pickle.load(f)

# with gzip.open("CCNCS_Internship/CNN/output_data/keyword_dict.pickle" , 'rb') as f:
#     keyword_dict = pickle.load(f)

# print("Train_x : " , train_x)
# print("\n")
# print("Train_y : " , train_y)
# print("\n")
# print("Keyword Dictionary : " , keyword_dict)
# print("\n")

# dim = train_x.shape[1]

# epochs = 100
# batch_size = 500
# k_n = 2

# kfold = StratifiedKFold(n_splits=k_n , shuffle = True , random_state=1)

# result = []

# try :
#     for train, test in kfold.split(train_x , train_y):
#         inputs = Input(shape = (dim , ) , name = "input")

#         embeddings_out = Embedding(input_dim = len(keyword_dict) , output_dim = 64 , name = "embedding")(inputs)

#         conv0 = Conv1D(32 , 3 , padding = 'same' , activation = 'relu')(embeddings_out)
#         pool0 = MaxPooling1D(pool_size = 2)(conv0)

#         conv1 = Conv1D(64 , 3 , padding = 'same' , activation = 'relu')(pool0)
#         pool1 = MaxPooling1D(pool_size = 2)(conv1)

#         flat = Flatten()(pool1)
#         dense = Dense(128 , activation = 'relu')(flat)

#         out = Dense(1 , activation = "sigmoid")(dense)

#         model = Model(inputs = [inputs] , outputs = out)
#         model.compile(optimizer = RMSprop() , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#         model_dir = './CNN_output_data'

#         if not os.path.exists(model_dir):
#             os.mkdir(model.dir)
#         model_path = model_dir + '/CNN.keras'
#         early_stopping = EarlyStopping(monitor = 'val_accuracy' , patience = 3)

#         hist = model.fit(x = train_x[train] , y = train_y[train] , batch_size=batch_size , epochs=epochs , validation_data=(train_x[test] , train_y[test]) , callbacks=[early_stopping])
#         eva = model.evaluate(x=train_x[test], y=train_y[test], batch_size=batch_size)
#         print("evaluation : ", eva)
#         result.append(eva)
#     model.save(model_dir + './CNN_rcs_test_1.model')
#     model.summary()

#     print("cross : " , result)
    
# except Exception as e:
#     print(f"Error during K-Fold Cross-Validation: {e}")
#     inputs = Input(shape=(dim,), name='input')
    
#     # CNN Architecture
#     embeddings_out = Embedding(input_dim=len(keyword_dict), output_dim=64, name='embedding')(inputs)
    
#     conv0 = Conv1D(32, 3, padding='same', activation='relu')(embeddings_out)
#     pool0 = MaxPooling1D(pool_size=2)(conv0)
    
#     conv1 = Conv1D(64, 3, padding='same', activation='relu')(pool0)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)
    
#     flat = Flatten()(pool1)
#     dense = Dense(128, activation='relu')(flat)
    
#     out = Dense(1, activation='sigmoid')(dense)

#     model = Model(inputs=[inputs], outputs=out)
#     model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

#     model_dir = './CNN_output_data'
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     model_path = model_dir + "/CNN.keras"
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
#     checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_accuracy", mode='max', verbose=1, save_best_only=True)            
#     hist = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping])

#     model.summary()









# FINAL.
import os
import pandas as pd
import numpy as np
import pickle
import gzip
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

with gzip.open("CCNCS_Internship/CNN/output_data/train_x.pickle" , 'rb') as f:
    train_x = pickle.load(f)

with gzip.open("CCNCS_Internship/CNN/output_data/train_y.pickle" , 'rb') as f:
    train_y = pickle.load(f)

with gzip.open("CCNCS_Internship/CNN/output_data/keyword_dict.pickle" , 'rb') as f:
    keyword_dict = pickle.load(f)



dim = train_x.shape[1]

epochs = 10
batch_size = 500
k_n = 5

kfold = StratifiedKFold(n_splits=k_n, shuffle=True, random_state=1)

result = []



inputs = Input(shape=(train_x.shape[1],), name='input')
embeddings_out = Embedding(input_dim=len(keyword_dict), output_dim=64, name='embedding')(inputs)

conv0 = Conv1D(32, 1, padding='same')(embeddings_out)
pool0 = MaxPooling1D(pool_size=1)(conv0)
flat = Flatten()(pool0)
out = Dense(1, activation='sigmoid')(flat)

model = Model(inputs=[inputs], outputs=out)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model_dir = './CNN_output_data'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = model_dir + "/CNN.keras"
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_accuracy", mode='max', verbose=1, save_best_only=True)

hist = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping])

model.summary()
print("done")







# import os
# import pandas as pd
# import numpy as np
# import pickle
# import gzip
# import tensorflow as tf

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.utils import resample

# os.environ["CUDA_VISIBLE_DEVICES"] = ''
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)


# with gzip.open("CCNCS_Internship/CNN/output_data/train_x.pickle" , 'rb') as f:
#     train_x = pickle.load(f)

# with gzip.open("CCNCS_Internship/CNN/output_data/train_y.pickle" , 'rb') as f:
#     train_y = pickle.load(f)

# with gzip.open("CCNCS_Internship/CNN/output_data/keyword_dict.pickle" , 'rb') as f:
#     keyword_dict = pickle.load(f)


# dim = train_x.shape[1]

# epochs = 100
# batch_size = 500

# # Ensure at least 2 samples per class
# unique_classes, class_counts = np.unique(train_y, return_counts=True)
# for cls, count in zip(unique_classes, class_counts):
#     if count < 2:
#         indices = np.where(train_y == cls)[0]
#         n_samples_needed = 2 - count
#         resampled_indices = resample(indices, replace=True, n_samples=n_samples_needed, random_state=1)
#         train_x = np.concatenate([train_x, train_x[resampled_indices]], axis=0)
#         train_y = np.concatenate([train_y, train_y[resampled_indices]], axis=0)

# # Verify class distribution
# unique_classes, class_counts = np.unique(train_y, return_counts=True)
# print(f"Class counts after resampling: {dict(zip(unique_classes, class_counts))}")

# # Use cross-validation for small datasets
# if len(train_y) < 10:
#     print("Using cross-validation due to small dataset size.")

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#     results = []

#     for train_index, test_index in skf.split(train_x, train_y):
#         x_train_fold, x_test_fold = train_x[train_index], train_x[test_index]
#         y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]

#         inputs = Input(shape=(dim,), name='input')

#         # CNN Architecture
#         embeddings_out = Embedding(input_dim=len(keyword_dict), output_dim=64, name='embedding')(inputs)

#         conv0 = Conv1D(32, 3, padding='same', activation='relu')(embeddings_out)
#         pool0 = MaxPooling1D(pool_size=2)(conv0)

#         conv1 = Conv1D(64, 3, padding='same', activation='relu')(pool0)
#         pool1 = MaxPooling1D(pool_size=2)(conv1)

#         flat = Flatten()(pool1)
#         dense = Dense(128, activation='relu')(flat)

#         out = Dense(1, activation='sigmoid')(dense)

#         model = Model(inputs=[inputs], outputs=out)
#         model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

#         early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
#         model_dir = './CNN_output_data'
#         if not os.path.exists(model_dir):
#             os.mkdir(model_dir)
#         model_path = model_dir + "/CNN.keras"
#         checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_accuracy", mode='max', verbose=1, save_best_only=True)

#         hist = model.fit(x_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, validation_data=(x_test_fold, y_test_fold), callbacks=[checkpoint, early_stopping])

#         scores = model.evaluate(x_test_fold, y_test_fold, verbose=0)
#         results.append(scores)

#     print("Cross-validation results:", results)
# else:
#     # Proceed with train/test split if the dataset is sufficiently large
#     train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1, stratify=train_y)

#     inputs = Input(shape=(dim,), name='input')

#     # CNN Architecture
#     embeddings_out = Embedding(input_dim=len(keyword_dict), output_dim=64, name='embedding')(inputs)

#     conv0 = Conv1D(32, 3, padding='same', activation='relu')(embeddings_out)
#     pool0 = MaxPooling1D(pool_size=2)(conv0)

#     conv1 = Conv1D(64, 3, padding='same', activation='relu')(pool0)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     flat = Flatten()(pool1)
#     dense = Dense(128, activation='relu')(flat)

#     out = Dense(1, activation='sigmoid')(dense)

#     model = Model(inputs=[inputs], outputs=out)
#     model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

#     model_dir = './CNN_output_data'
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     model_path = model_dir + "/CNN.keras"
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
#     checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_accuracy", mode='max', verbose=1, save_best_only=True)

#     hist = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), callbacks=[checkpoint, early_stopping])

#     model.summary()

# # Save the final model
# model.save(model_dir + '/CNN.keras')
