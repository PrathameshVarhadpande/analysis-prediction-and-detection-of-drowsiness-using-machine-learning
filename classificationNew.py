from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D,Conv2D
import tensorflow as tf
from tensorflow import keras
# from keras.optimizers import adam_v2
from keras.layers import Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam

df = pd.read_csv('totalwithmaininfo2.csv',sep=',')
#df = df.drop(df.columns[0],axis=1)

# print(df.columns.to_list())

# y = df['Y'].values
# X = df.drop(['Y'], axis = 1)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state=30)

train_percentage = 17/22
train_index = int(len(df)*train_percentage)
test_index = len(df)-train_index
df_train = df[:train_index]
df_test = df[-test_index:]
X_test = df_test.drop(["Y"],axis=1)
y_test = df_test["Y"]
X_train = df_train.drop('Y',axis=1)
y_train = df_train['Y']

X_train_shaped = np.expand_dims(X_train, axis=2)
X_train_shaped.shape
X_test_shaped = np.expand_dims(X_test, axis=2)
X_test_shaped.shape

## Create Model ##

model = Sequential()

model.add(Conv1D(64,kernel_size = 3, activation = 'relu', input_shape = (8,1),data_format='channels_last'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
#model.add(Dense(1,activation = 'sigmoid'))


## Compile Model ##
#optimizer = Adam(lr=0.00001)
#adam = Adam(learning_rate=0.001, name='Adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ## Train Model and Check Validation Accuracy ##
model.fit(X_train_shaped, y_train, validation_data = (X_test_shaped,y_test), epochs = 20)

model.summary()

# # pred_cnn = model.predict_classes(X_test_shaped)
# # pred_cnn = average(pred_cnn)
# # y_score_7 = model.predict_proba(X_test_shaped)
# # acc7 = accuracy_score(y_test, np.array(pred_cnn))
# # f1_score_7 = metrics.f1_score(y_test, pred_cnn)
# # roc_7 = metrics.roc_auc_score(y_test, y_score_7)
# # print([acc7,f1_score_7,roc_7])
# # print(confusion_matrix(y_test, pred_cnn))
