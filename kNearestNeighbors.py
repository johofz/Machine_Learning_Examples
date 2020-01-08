import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = pd.read_csv('Practice_Data/breast-cancer-wisconsin.data')

drop_na = True
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

count = 10000
predictions = np.zeros((1,2))
accuracys = 0

for i in range(count):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    from_pickle = False

    if from_pickle:
        pickle_in = open('KNeighborsClassifier.pickle', 'rb')
        clf = pickle.load(pickle_in)
    else:
        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)
        with open('KNeighborsClassifier.pickle', 'wb') as f:
            pickle.dump(clf, f)


    accuracys += clf.score(X_test, y_test)

    data_point = np.array([[10, 3, 2, 1, 2, 3, 5, 1, 1]])
    prediction = clf.predict_proba(data_point)
    predictions += prediction

prediction = predictions / count
accuracy = accuracys / count
print(f'Accuracy: {accuracy}')
print(f'prediction: {prediction}')