# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:36:48 2019
@author: Rukon
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# Importing the dataset
dataset = pd.read_csv('car.csv')

X = dataset.iloc[:, :-1 ].values
y = dataset.iloc[:, 6].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
X[:, 1] = labelEncoder_X.fit_transform(X[:, 1])
X[:, 2] = labelEncoder_X.fit_transform(X[:, 2])
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
X[:, 4] = labelEncoder_X.fit_transform(X[:, 4])
X[:, 5] = labelEncoder_X.fit_transform(X[:, 5])
X = X.astype(float)
y = y.astype('U')

"""
oneHotEncoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5])
X = oneHotEncoder.fit_transform(X).toarray()
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

# finding the optimal value of K in KNN
from sklearn.neighbors import KNeighborsClassifier
scores = []
k_range = range(1,26)
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
print(scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

#Fitting classifier to the Training set
classifier = KNeighborsClassifier(n_neighbors = 6)
classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict([X_test[0]])
y_pred = classifier.predict(X_test)

# Evaluating the performance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

"""
#plotting the confusion matrix
import seaborn as sns 
index = ['acc','good','unacc', 'vgood']  
columns = ['acc','good','unacc', 'vgood'] 
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, cmap="YlGnBu", annot=True)
"""

#plotting the confusion matrix 2
import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, cmap="YlGnBu", annot=True, ax = ax) #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['accepted','good','unaccepted', 'vgood'])
ax.yaxis.set_ticklabels(['accepted','good','unaccepted', 'vgood'])

#plotting classification report
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def plot_classification_report(y_tru, y_prd, figsize=(10, 6), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                cmap="YlGnBu",
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax)

plot_classification_report(y_test, y_pred)


#saving the model
#filename = 'car_evaluation_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))



