import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#test dataset
xtest=[[0.793553,16,10,906,556]]


data = pd.read_csv("data2.csv")
X = data.drop('label', axis=1)
y = data['label']


#Data processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)


#Algorithm comparison
algorithms = {"SVM Classifier:":SVC(kernel='linear'),"RandomForestClassifier":RandomForestClassifier(n_estimators=100, random_state=23),"KNeighborsClassifier":KNeighborsClassifier(n_neighbors=23),"Logistic regression Classifier": linear_model.LogisticRegression(),"naivbayes Classifier": GaussianNB()
}

results = {}
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

best_algo = max(results, key=results.get)
print('\nBest Algorithm is %s with a %f %%' % (best_algo, results[best_algo]*100))

classifier = algorithms[best_algo]
classifier.fit(X_train, y_train)
y_pred = classifier.predict(xtest)
print(y_pred)
