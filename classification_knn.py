from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#set plot style
plt.style.use('ggplot')

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)

    # Try KNN with 5 neighbors
    knn = KNeighborsClassifier()

    # Fit training data
    knn.fit(X_train, y_train)

    #Cneck Accuracy Score
    print('Default Accuracy: {}'.format(round(knn.score(X_test, y_test), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)

        # Fitting
        knn.fit(X_train, y_train)

        # Training Accuracy
        train_accuracy[acc] = knn.score(X_train, y_train)

        # Testing Accuracy
        test_accuracy[acc] = knn.score(X_test, y_test)

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

    plot_confusion_matrix(knn,X_test,y_test)
    plt.grid(False)
    plt.savefig("static/graphs/cf2.png")
    plt.show()

if __name__ == '__main__':
    df = loadFile('drowsiness_dataset.csv')
    minorEDA(df)
    x, y = feature('Y', df)
    TestTrainFitPlot(x, y)