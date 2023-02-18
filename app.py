from flask import Flask, jsonify, request
from flask_restful import Resource, Api
#import BTC_prediction
from BTC_prediction import prenext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from flask import Flask, request, jsonify

scaler = MinMaxScaler()
model = load_model('BTCprd.h5')

app = Flask(__name__)

@app.route('/predict',  methods=['POST'])
def predict_price():
    predicted_price = prenext()
    return f'Predicted price: {predicted_price}'
    '''model_input = scaler.transform(last_price)
    model_input = np.reshape(model_input, (1, 1, 1))
    prediction = model.predict(model_input)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    # return the predicted price
    return f'Predicted price: {predicted_price}'
'''

if __name__ == '__main__':
    app.run(debug=True)