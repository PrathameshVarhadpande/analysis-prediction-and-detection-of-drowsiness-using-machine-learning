#Logistic Regression

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns

df = pd.read_csv('totalwithallinfo.csv')

df.drop_duplicates(inplace=True)

# df.drop('Y', axis=1, inplace=True)

X = df.iloc[:,df.columns != 'Y']
y = df['Y']

X_train1, X_test1, y_train1, y_test1 = train_test_split(
X, y, test_size=0.20, random_state=5, stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train1)
X_train_scaled = scaler.transform(X_train1)

X_test_scaled = scaler.transform(X_test1)

model = LogisticRegression()

model.fit(X_train_scaled, y_train1)

cal1 = model.decision_function(X_test1)

print("CAL_1 = ",cal1)

y_pred = model.predict(X_test_scaled)

train_acc = model.score(X_train_scaled, y_train1)
print("The Accuracy for Training Set is {}".format(train_acc*100))


test_acc = accuracy_score(y_test1, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))


print(classification_report(y_test1, y_pred))


cm=confusion_matrix(y_test1,y_pred)


#KNN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#set plot style
plt.style.use('ggplot')

y_test2_1 = 0
cal2_1 = 0

def loadFile(path):
    #Load Excel File into Pandas DataFrame
    df = pd.read_csv(path)
    return df

def minorEDA(df):
    lineBreak = '------------------'

    #Check Shape
    print(lineBreak*3)
    print("Shape:")
    print(df.shape)
    print(lineBreak*3)
    #Check Feature Names
    print("Column Names")
    print(df.columns)
    print(lineBreak*3)
    #Check types, missing, memory
    print("Data Types, Missing Data, Memory")
    print(df.info())
    print(lineBreak*3)

def feature(feature, df):
    # Create arrays for the features and the response variable
    y = df[feature]
    x = df.drop(feature, axis=1)
    return x, y

def TestTrainFitPlot(X, y):
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Split into training and test set
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

    y_test2_1 = y_test2

    # Try KNN with 5 neighbors
    knn = KNeighborsClassifier()

    # Fit training data
    knn.fit(X_train2, y_train2)

    #Cneck Accuracy Score
    print('Default Accuracy: {}'.format(round(knn.score(X_test2, y_test2), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)

        # Fitting
        knn.fit(X_train2, y_train2)

        cal2 = knn.predict_proba(X_test2)

        cal2_1 = cal2

        # Training Accuracy
        train_accuracy[acc] = knn.score(X_train2, y_train2)

        # Testing Accuracy
        test_accuracy[acc] = knn.score(X_test2, y_test2)

    #Plotting
    #Set Main Title
    plt.title('KNN Neighbors')
    #Set X-Axis Label
    plt.xlabel('Neighbors\n(#)')
    #Set Y-Axis Label
    plt.ylabel('Accuracy\n(%)', rotation=0, labelpad=35)
    #Place Testing Accuracy
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    #Place Training Accuracy
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    #Append Labels on Testing Accuracy
    for a,b in zip(neighbors, test_accuracy):
        plt.text(a, b, str(round(b,2)))
    #Add Legend
    plt.legend()
    #Generate Plot
    plt.show()

if __name__ == '__main__':
    df = loadFile('totalwithmaininfo.csv')
    minorEDA(df)
    x, y = feature('Y', df)
    TestTrainFitPlot(x, y)

#Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("totalwithmaininfo.csv")

dataset.shape

dataset.head()

X = dataset.drop('Y', axis=1)
y = dataset['Y']

from sklearn.model_selection import train_test_split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.20)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train3, y_train3)

cal3 = classifier.predict_proba(X_test3)

y_pred = classifier.predict(X_test3)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test3, y_pred))
print(classification_report(y_test3, y_pred))

#Random Forest

import pandas as pd
import numpy as np

dataset = pd.read_csv('totalwithmaininfo.csv')

dataset.head()

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split

X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train4 = sc.fit_transform(X_train4)
X_test4 = sc.transform(X_test4)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train4, y_train4)

#cal4 = regressor.predict_proba(X_test4)

y_pred = regressor.predict(X_test4)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test4,y_pred))
print(classification_report(y_test4,y_pred))
print(accuracy_score(y_test4, y_pred))

#Support Vector Machine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("totalwithmaininfo.csv")

bankdata.shape

bankdata.head()

X = bankdata.drop('Y', axis=1)
y = bankdata['Y']

from sklearn.model_selection import train_test_split
X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train5, y_train5)

#cal5 = svclassifier.predict_proba(X_test5)

y_pred = svclassifier.predict(X_test5)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test5,y_pred))
print(classification_report(y_test5,y_pred))


#ML Classification Graph and Calibration Curve

import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Logistic Regression':64, 'KNN':82, 'Decision Tree':78, 'Random Forest':100, 'SVM':64}
courses = list(data.keys())
values = list(data.values())
  
colors=['lightcoral','seagreen','orchid','royalblue','darkorange']

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.title('Comparitive Analysis on Drowsiness Dataset using ML Algorithms')
plt.grid(True)
plt.ylabel('Accuracy %')
plt.xlabel('Machine Learning Algorithms')

plt.bar(courses,values,color=colors)
plt.savefig('graphs/mlclassification.png')
plt.show()


# Calibration Curve

from sklearn.calibration import calibration_curve
plt.figure(figsize=(10,5))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
fraction_of_positives, mean_predicted_value=calibration_curve(y_test1,cal1,n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives,"s-",
                 label="%s" % 'Logistic Regression')
fraction_of_positives, mean_predicted_value=calibration_curve(y_test2_1,cal2_1,n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives,"s-",
                 label="%s" % 'KNN')
fraction_of_positives, mean_predicted_value=calibration_curve(y_test3,cal3,n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives,"s-",
                 label="%s" % 'Decision Tree')
# fraction_of_positives, mean_predicted_value=calibration_curve(y_test4,cal3,n_bins=10)
# plt.plot(mean_predicted_value, fraction_of_positives,"s-",
#                  label="%s" % 'Random Forest')
# fraction_of_positives, mean_predicted_value=calibration_curve(y_test5,cal5,n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives,"s-",
                 label="%s" % 'SVM')

plt.legend(loc="lower right")
plt.title('Calibration Curve on Classification Models')
plt.grid(True)
plt.savefig('graphs/calib2.png')
plt.show()



