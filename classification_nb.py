import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

bankdata = pd.read_csv("drowsiness_dataset.csv")

bankdata.shape

bankdata.head()

X = bankdata.iloc[:, 0:8].values
y = bankdata.iloc[:, 0].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=69)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

import numpy as np

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))

plot_confusion_matrix(model,X_test,y_test)
plt.grid(False)
plt.savefig("static/graphs/cf3.png")
plt.show()