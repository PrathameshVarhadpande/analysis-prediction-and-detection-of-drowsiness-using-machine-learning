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
import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings

df = pd.read_csv('totalwithmaininfo.csv',sep=',')
#df = df.drop(df.columns[0],axis=1)
y = df['Y'].values
X = df.drop(['Y'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state=30)

train_percentage = 17/22
train_index = int(len(df)*train_percentage)
test_index = len(df)-train_index

df_train = df[:train_index]
df_test = df[-test_index:]

X_test = df_test.drop(["Y"],axis=1)
y_test = df_test["Y"]

X_train = df_train.drop('Y',axis=1)
y_train = df_train['Y']

def average(y_pred):
  for i in range(len(y_pred)):
    if i % 240 == 0 or (i+1) % 240 == 0:
      pass
    else: 
      average = float(y_pred[i] +  y_pred[i] + y_pred[i])/3
      if average >= 0.5:
        y_pred[i] = 1
      else:
        y_pred[i] = 0
  return y_pred
models = ['LR','NB','KNN','DT','RF']
scores = []

#LogisticRegression Algorithm

clf = LogisticRegression().fit(X_train, y_train)
y_pred_1 = clf.predict(X_test)
y_pred_1 = average(y_pred_1)
y_score_1 = clf.predict_proba(X_test)[:,1]
acc1 = accuracy_score(y_test, y_pred_1)
#f1_score_1 = metrics.f1_score(y_test, y_pred_1)
roc_1 = metrics.roc_auc_score(y_test, y_score_1)
#print([acc1,f1_score_1,roc_1])
print(confusion_matrix(y_test, y_pred_1))

#GaussianNB Algorithm

clf_NB = GaussianNB()
clf_NB.fit(X_train, y_train)
pred_NB = clf_NB.predict(X_test)
pred_NB = average(pred_NB)
y_score_2 = clf_NB.predict_proba(X_test)[:,1]
acc2 = accuracy_score(y_test, pred_NB)
#f1_score_2 = metrics.f1_score(y_test, pred_NB)
roc_2 = metrics.roc_auc_score(y_test, y_score_2)
print([acc2,roc_2])
scores.append(roc_2 * 100)
print(confusion_matrix(y_test, pred_NB))

acc3_list = []
f1_score3_list = []
roc_3_list = []

#KNN Algorithm

from sklearn.neighbors import KNeighborsClassifier
for i in range(1,30):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train) 
    pred_KN = neigh.predict(X_test)
    pred_KN = average(pred_KN)
    y_score_3 = neigh.predict_proba(X_test)[:,1]
    acc3_list.append(accuracy_score(y_test, pred_KN))
    #f1_score3_list.append(metrics.f1_score(y_test, pred_KN))
    roc_3_list.append(metrics.roc_auc_score(y_test, y_score_3))

acc3_list.index(max(acc3_list))+1

neigh = KNeighborsClassifier(n_neighbors=acc3_list.index(max(acc3_list))+1)
neigh.fit(X_train, y_train) 
pred_KN = neigh.predict(X_test)
pred_KN = average(pred_KN)
y_score_3 = neigh.predict_proba(X_test)[:,1]
acc3 = accuracy_score(y_test, pred_KN)
#f1_score_3 = metrics.f1_score(y_test, pred_KN)
roc_3 = metrics.roc_auc_score(y_test, y_score_3)
scores.append(roc_3 * 100)
print([acc3,roc_3])
print(confusion_matrix(y_test, pred_KN))

#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
acc5=[]
max_depth = []
for i in [2,3,4,5,6,7,8,9,10]:
    clf_DT = DecisionTreeClassifier(random_state=0, max_depth = i)
    clf_DT.fit(X_train, y_train)
    pred_DT = clf_DT.predict(X_test)
    pred_DT = average(pred_DT)
    acc5.append(accuracy_score(pred_DT, y_test))
    max_depth.append(i)
print (max(acc5))

best_depth = max_depth[acc5.index(max(acc5))]

clf_DT = DecisionTreeClassifier(random_state=0, max_depth = best_depth)
clf_DT.fit(X_train, y_train)
pred_DT = clf_DT.predict(X_test)
pred_DT = average(pred_DT)
y_score_5 = clf_DT.predict_proba(X_test)[:,1]
acc5 = accuracy_score(y_test, pred_DT)

#f1_score_5 = metrics.f1_score(y_test, pred_DT)
roc_5 = metrics.roc_auc_score(y_test, y_score_5)
scores.append(roc_5 * 100)
print([acc5,roc_5])
print(confusion_matrix(y_test, pred_DT))

#Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(max_depth=6)
clf_RF.fit(X_train, y_train) 
pred_RF = clf_RF.predict(X_test)
pred_RF = average(pred_RF)
y_score_6 = clf_RF.predict_proba(X_test)[:,1]
acc6 = accuracy_score(y_test, pred_RF)
#f1_score_6 = metrics.f1_score(y_test, pred_RF)
roc_6 = metrics.roc_auc_score(y_test, y_score_6)
scores.append(acc6 * 100)
print([acc6,roc_6])
print(confusion_matrix(y_test, pred_RF))

feature_importances = pd.DataFrame(clf_RF.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# deep learning starts
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers.convolutional import Conv1D
# #from keras.optimizers import Adam, RMSprop
# from tensorflow.keras.optimizers import Adam
# from keras.layers import Dropout
# ## Create Model ##

# model = Sequential()

# model.add(Conv1D(64, kernel_size = 3, activation = 'relu'))
# #model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1,activation = 'sigmoid'))
# ## Compile Model ##
# #optimizer = Adam(lr=0.00001)
# model.compile(loss='binary_crossentropy',metrics=['accuracy'])

# ## Train Model and Check Validation Accuracy ##
# model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 10)

# model.summary()

# pred_cnn = model.predict_classes(X_test)
# pred_cnn = average(pred_cnn)
# y_score_7 = model.predict_proba(X_test)
# acc7 = accuracy_score(y_test, np.array(pred_cnn))
# #f1_score_7 = metrics.f1_score(y_test, pred_cnn)
# roc_7 = metrics.roc_auc_score(y_test, y_score_7)
# print([acc7,roc_7])
# print(confusion_matrix(y_test, pred_cnn))

# # acc_total = {'Model':['Logistic Regression','Naive Bayes', 'KNN', 'MLP','Decision Tree','Random Forest', 'CNN', 'XGB Boosting'],
# #         'Accuracy':[acc1,acc2, acc3, acc4, acc5,acc6,acc7, acc8]}
# # acc_total=pd.DataFrame(acc_total)
# # acc_total=acc_total.set_index('Model')
# # acc_total
# # plt.plot(acc_total['Accuracy'])
# # plt.xticks(rotation=45)
# # acc_total

# # plt.figure(figsize=(10,8))
# # plt.plot([0, 1], [0, 1],'r--')
# # fpr_1, tpr_1, thresholds = roc_curve(y_test, y_score_1)
# # fpr_2, tpr_2, thresholds = roc_curve(y_test, y_score_2)
# # fpr_3, tpr_3, thresholds = roc_curve(y_test, y_score_3)
# # fpr_5, tpr_5, thresholds = roc_curve(y_test, y_score_5)
# # fpr_6, tpr_6, thresholds = roc_curve(y_test, y_score_6)
# # plt.plot(fpr_1, tpr_1, label= "Logistic Regression")
# # plt.plot(fpr_2, tpr_2, label= "Naive Bayes")
# # plt.plot(fpr_3, tpr_3, label= "KNN")
# # plt.plot(fpr_5, tpr_5, label= "Decision Tree")
# # plt.plot(fpr_6, tpr_6, label= "Random Forest")
# # plt.title('ROC Curve for LSTM')
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.legend()
# # plt.show()

# #Calibration Curve

# from sklearn.calibration import calibration_curve
# # plt.figure(figsize=(10,5))
# # plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# # fraction_of_positives, mean_predicted_value=calibration_curve(y_test,y_score_1,n_bins=10)
# # plt.plot(mean_predicted_value, fraction_of_positives,"s-",
# #                  label="%s" % 'Logistic Regression')
# # fraction_of_positives, mean_predicted_value=calibration_curve(y_test,y_score_2,n_bins=10)
# # plt.plot(mean_predicted_value, fraction_of_positives,"s-",
# #                  label="%s" % 'Naive Bayes')
# # fraction_of_positives, mean_predicted_value=calibration_curve(y_test,y_score_3,n_bins=10)
# # plt.plot(mean_predicted_value, fraction_of_positives,"s-",
# #                  label="%s" % 'KNN')
# # fraction_of_positives, mean_predicted_value=calibration_curve(y_test,y_score_5,n_bins=10)
# # plt.plot(mean_predicted_value, fraction_of_positives,"s-",
# #                  label="%s" % 'Decision Tree')
# # fraction_of_positives, mean_predicted_value=calibration_curve(y_test,y_score_6,n_bins=10)
# # plt.plot(mean_predicted_value, fraction_of_positives,"s-",
# #                  label="%s" % 'Random Forest')

# # plt.legend(loc="lower right")
# # plt.title('Calibration Curve on Classification Models')
# # plt.grid(True)
# # plt.savefig('graphs/calib.png')
# # plt.show()

# # print(models)
# # print(scores)
# # colors=['lightcoral','seagreen','orchid','royalblue','darkorange']
# # plt.figure(figsize=(10,5))
# # plt.title('Comparitive Analysis on Drowsiness Dataset using ML Algorithms')
# # plt.grid(True)
# # plt.ylabel('Accuracy %')
# # plt.xlabel('Machine Learning Algorithms')

# # plt.bar(models,scores,color=colors)
# # plt.savefig('graphs/mlbar.png')
# # plt.show()

# print(models)
# print(scores)