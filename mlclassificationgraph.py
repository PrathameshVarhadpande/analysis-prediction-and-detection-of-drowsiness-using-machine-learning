import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Logistic Regression':77, 'KNN':97.5, 'Naive Bayes':79, 'SVM':87}
courses = list(data.keys())
values = list(data.values())
  
colors=['lightcoral','seagreen','orchid','royalblue']

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.title('Comparitive Analysis on Drowsiness Dataset using ML Algorithms')
plt.grid(True)
plt.ylabel('Accuracy %')
plt.xlabel('Machine Learning Algorithms')

plt.bar(courses,values,color=colors)
plt.savefig('static/graphs/mlclassification2.png')
plt.show()


