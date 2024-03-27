import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data2.csv")

X = data.drop('label', axis=1)
print(X)####----data----####

y = data['label']
print(y)####--------label----####


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

rflassifier = RandomForestClassifier(n_estimators=100, random_state=23)
rflassifier.fit(X_train, y_train)
score = rflassifier.score(X_test, y_test)
print(score*100)

xtest=[[0.793553,16,10,906,556]]##-----test data---###
print(xtest)

y_pred = rflassifier.predict(xtest)
print(y_pred)



