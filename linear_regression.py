import math
import pandas as pd
import quandl
import numpy as np
from api_data import api_key
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = api_key
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT',  'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
df['lable'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['lable'], 1))
y = np.array(df['lable'])
X = preprocessing.scale(X)
y = np.array(df['lable'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

algorithm = 'svm'

if algorithm == 'svm':
    clf = svm.SVR()
elif algorithm == 'linear':
    clf = LinearRegression()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)

print(df.head())