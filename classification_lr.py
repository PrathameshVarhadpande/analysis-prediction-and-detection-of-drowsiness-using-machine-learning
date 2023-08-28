import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

from sklearn.metrics import plot_confusion_matrix

import seaborn as sns

df = pd.read_csv('drowsiness_dataset.csv')

df.drop_duplicates(inplace=True)

# df.drop('Y', axis=1, inplace=True)

X = df.iloc[:,df.columns != 'Y']
y = df['Y']

X_train, X_test, y_train, y_test1 = train_test_split(
X, y, test_size=0.20, random_state=5, stratify=y)

print("Y:test = ",y_test1)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

cal1 = model.decision_function(X_test)

y_pred = model.predict(X_test_scaled)

train_acc = model.score(X_train_scaled, y_train)
print("The Accuracy for Training Set is {}".format(train_acc*100))


test_acc = accuracy_score(y_test1, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))


print(classification_report(y_test1, y_pred))


cm=confusion_matrix(y_test1,y_pred)

plot_confusion_matrix(model,X_test,y_test1)
plt.savefig("static/graphs/cf1.png")
plt.show()


