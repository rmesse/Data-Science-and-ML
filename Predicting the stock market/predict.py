# Libraries
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Data import
df = pd.read_csv('sphist.csv')

# Conversion of date column to datetime format
df['Date']=pd.to_datetime(df['Date'])

# Data sorted in ascending order of date 
df = df.sort_values(by=['Date'])

# Computation of indicators
df['mean_5days'] = df["Close"].rolling(window = 5).mean().shift(1)
df['mean_30days'] = df["Close"].rolling(window = 30).mean().shift(1)
df['mean_365days'] = df["Close"].rolling(window = 365).mean().shift(1)
df['ratio_mean'] = df['mean_5days']/df['mean_365days']
df['std_5days'] = df["Close"].rolling(window = 5).std().shift(1)
df['std_30days'] = df["Close"].rolling(window = 30).std().shift(1)
df['std_365days'] = df["Close"].rolling(window = 365).std().shift(1)
df['ratio_std'] = df['std_5days']/df['std_365days']

# Removal of rows with dates before 1951-01-03
df = df[df["Date"] > dt.datetime(year=1951, month=1, day=2)]

# Removal of rows with `NaN` values
df = df.dropna()

# Train and test data split
train = df[df["Date"] < dt.datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= dt.datetime(year=2013, month=1, day=1)]

# Linear regression model
lr = LinearRegression()
features = ['mean_5days','mean_30days','mean_365days',
           'std_5days','std_30days','std_365days',
           'ratio_mean','ratio_std']
lr.fit( train[ features ] , train['Close'] )
predictions = lr.predict(test[features])
MAE = mean_absolute_error(test['Close'],predictions)
print(MAE)

