# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger



app= Flask(__name__)
Swagger(app)
pickle_in = open('classifier.pkl','rb')
classifier= pickle.load(pickle_in)

@app.route('/')
def Welcome():
    return "Welcome all"
@app.route('/predict', methods=['Get'])
def predict_note_authentication():
    """let's authenticate the Bank notes
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: the output values
    

    

    """
    variance= request.args.get('variance')
    skewness= request.args.get('skewness')
    curtosis= request.args.get('curtosis')
    entropy= request.args.get('entropy')
    prediction= classifier.predict([[variance,skewness,curtosis, entropy]])
    return "the prediction is" + str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    df= pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df)
   
    return "the prediction is" + str(list(prediction))

if __name__=='__main__':
    app.run()