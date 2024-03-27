import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("data2.csv")
X = data.drop('label', axis=1)
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
score = svclassifier.score(X_test, y_test)
print(score*100)

xtest=[[0.793553,16,10,906,556]]

y_pred = svclassifier.predict(xtest)
print(xtest)
print(y_pred)



