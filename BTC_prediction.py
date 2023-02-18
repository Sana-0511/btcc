#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

import yfinance as yf

yf.pdr_override()


# In[12]:


#Defining data
crypto_curr = 'BTC'
actual_curr = 'USD'

start = dt.datetime(2019,1,1)
end = dt.datetime.now()

# Defining Training Set
training_set = pdr.get_data_yahoo(f'{crypto_curr}-{actual_curr}', start = start, end = end)

print("Overview of training set: \n",training_set.head())
print("\nSize of Training set:",len(training_set))


# In[21]:


# Normalization of training set
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(training_set['Close'].values.reshape(-1,1))
print("The training_set[Close] normalised between 0-1:\n",scaled_data)


# In[25]:


# Length of the Data set
training_set_size = len(scaled_data) - 1

X_train = scaled_data[0:training_set_size]
Y_train = scaled_data[1:training_set_size + 1]

X_train


# In[27]:


#Creating neural network
model = Sequential()

model.add(LSTM(units=50, return_sequences = True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))    
model.add(Dense(units=1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, Y_train, batch_size = 32, epochs = 100)

model.save('BTCprd.h5')

# In[28]:


model.summary()


# In[30]:


# Testing data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()
test_set = pdr.get_data_yahoo(f'{crypto_curr}-{actual_curr}', start = test_start, end = test_end)
test_set.head()
print("Overview of test set: \n",test_set.head())
print("\nSize of Test set:",len(test_set))


# In[54]:


actual_prices = test_set['Close'].values
print("Actual BTC Prices :\n", actual_prices)

model_inputs = actual_prices.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)
print("\nModel Inputs :\n",model_inputs)

model_inputs = np.array(model_inputs)
model_inputs = np.reshape(model_inputs, (model_inputs.shape[0], 1, model_inputs.shape[1]))
prediction_prices = model.predict(model_inputs)
prediction_prices = scaler.inverse_transform(prediction_prices)
print("\nPredicted prices :\n",prediction_prices)


# In[71]:


import plotly.graph_objects as go

dates = pd.date_range(start='2020-01-01', end=pd.Timestamp.now(), freq='D')
date_array = dates.to_numpy()

prediction_price = prediction_prices.flatten()

# Garph

x1 = dates
y1 = actual_prices
x2 = dates
y2 = prediction_price

f1 = go.Figure(
    data = [
        go.Scatter(x=x1, y=y1, name="Actual Prices"),
        go.Scatter(x=x2, y=y2, name="Predicted Prices"),
    ],
    layout = {"xaxis": {"title": "Date"}, "yaxis": {"title": "Price"}, "title": "Price Prediction"}
)
f1


# In[78]:


# Predicting the next day
def lastp():
    last_price = actual_prices[-1]
    return last_price
def prenext():
    last_price = actual_prices[-1]
    last_price_scaled = scaler.transform(np.array([[last_price]]))
    last_price_scaled_reshaped = np.reshape(last_price_scaled, (1, 1, 1))

    predicted_price_scaled = model.predict(last_price_scaled_reshaped)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)


    prediction_date = dt.date.today() + dt.timedelta(days=1)
    print(f"Predicted Price for {prediction_date} :",predicted_price[0][0])
    return predicted_price[0][0]

prenext()





