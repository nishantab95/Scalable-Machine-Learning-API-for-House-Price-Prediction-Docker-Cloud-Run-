# imports 
import joblib
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('xgb_gbr_stacked_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = (np.array(list(data.values())).reshape(1, -1))
    new_data[0, -1] = np.log(new_data[0, -1]) 
    output = model.predict(new_data)
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)
