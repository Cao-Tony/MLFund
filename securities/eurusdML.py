import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

df = pd.read_csv('C:\\Users\\trung\\Documents\\MLFund\\MLFund\\dataFiles\\EURUSD=X (1).csv')

# view all rows in df
# pd.set_option('display.max+rows', df.shape[0]+1)
# pd.set_option('display.max_columns', df.shape[0]+1)

# drop any NaN rows
df = df.dropna()

x_label = df['Date']
y_label = df['Adj Close']

date_time = pd.to_datetime(df.pop('Date'), format='%Y.%m.%d')

plot_cols = ['Open', 'High', 'Low', 'Adj Close']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)


df_describe = df.describe().transpose()


