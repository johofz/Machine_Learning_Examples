import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from my_K_Nearest_Neighbors import myKNearestNeighbor, convert_data
from time import time

df = pd.read_csv('Practice_Data/breast-cancer-wisconsin.data')

drop_na = True
use_my_algorithm = True
df_size = len(df)

if drop_na:
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f'Lost Data: {df_size - len(df)}')
else:
    df.replace('?', -99999, inplace=True)

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

count = 10
predictions = np.zeros((1,2))
accuracys = 0
data_point = np.array([[10, 3, 2, 1, 2, 3, 5, 1, 1]])

start_time = time()
for i in range(count):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    from_pickle = False

    if from_pickle:
        pickle_in = open('KNeighborsClassifier.pickle', 'rb')
        clf = pickle.load(pickle_in)
        clf.fit(X_train, y_train)
    else:
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)
        with open('KNeighborsClassifier.pickle', 'wb') as f:
            pickle.dump(clf, f)            

    accuracys += clf.score(X_test, y_test)

    prediction = clf.predict_proba(data_point)
    predictions += prediction

prediction = predictions / count
accuracy = accuracys / count
print(f'time took sklearn: {time() - start_time}')
print(f'Accuracy sklearn: {accuracy}')
print(f'prediction sklearn: {prediction}')

accuracys = 0
start_time = time()
for i in range(count):
    train_data, test_data = convert_data(df, test_size=0.2)

    correct = 0
    total = 0

    for group in test_data:
        for data in test_data[group]:
            vote = myKNearestNeighbor(train_data, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    accuracys += correct / total

accuracy = accuracys / count
print(f'time took my K Nearest: {time() - start_time}')
print(f'Accuracy my K Nearest: {accuracy}')
