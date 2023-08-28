import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

dataset = pd.read_csv("drowsiness_dataset.csv")

dataset.shape

dataset.head()

X = dataset.drop('Y', axis=1)
y = dataset['Y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(classifier,X_test,y_test)
plt.grid(False)
#plt.savefig("static/graphs/cf3.png")
plt.show()