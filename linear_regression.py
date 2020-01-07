import pandas as pd
import quandl, math, datetime
import numpy as np
from api_data import api_key
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get("EOD/AAPL", authtoken=api_key)
print(df.head())

df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume',]]
df['HL_PCT'] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Low'] * 100.0
df['PCT_CHANGE'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100.0

df = df[['Adj_Close', 'HL_PCT',  'PCT_CHANGE', 'Adj_Volume']]

forecast_col = 'Adj_Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

algorithm = 'linear'

if algorithm == 'svm':
    clf = svm.SVR()
elif algorithm == 'linear':
    clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
    

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()