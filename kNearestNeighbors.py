import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = pd.read_csv('Practice_Data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

from_pickle = False

if from_pickle:
    pickle_in = open('KNeighborsClassifier.pickle', 'rb')
    clf = pickle.load(pickle_in)
else:
    clf = neighbors.KNeighborsClassifier()
    with open('KNeighborsClassifier.pickle', 'wb') as f:
        pickle.dump(clf, f)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
data_point = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
prediction = clf.predict_proba(data_point)
print(prediction)