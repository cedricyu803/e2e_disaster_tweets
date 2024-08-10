# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""


"""

#%% This file

"""
text pre-processing, feature extractions, engineering including encoding
pipeline

"""





#%% Preamble


import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# # import os
# # os.chdir(r'C:\Users\Cedric Yu\Desktop\Data Science\flask\nlp_disaster_tweets')

# #%% model

# load utils
from model_pipeline import *

# load pipelines and model
from joblib import load

pipeline_text = load("pipeline_text.joblib")
pipeline_model = load("pipeline_model.joblib")


#%% Flask webpage

# render_template loads templates
from flask import Flask, render_template, flash, redirect, url_for, request


# Flask class
# render_template by default loads templates from 'templates' folder in the root level of the app; change template folder using template_folder
app = Flask(__name__)


# create a secure key to prevent modifying cookies and attacks
# create a random string with secrets
# import secrets
# secrets.token_hex(16)
app.config['SECRET_KEY'] = 'b856782614d9383f8e591c833996fc75'



# the route() decorator tells Flask what URL should trigger our function
# http://127.0.0.1:5000/
# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

# get the data for the requested query
@app.route('/success/<name>')
def success(name):
    
    test = str(name)
    
    X_test = pd.DataFrame([test], columns = ['text'])
    X_test_text = text_preprocess(X_test)
    X_test_textp = pipeline_text.transform(X_test_text).toarray()
    if pipeline_model.predict(X_test_textp)[0] == 0:
        message = 'Not a disaster.'
    else: 
        message = 'A disaster!'
    # return test
    return "<xmp>" + message + " </xmp> "
    


# # print predictions
# @app.route('/success')
# def success(name):
    
#     test = str(name)
    
#     X_test = pd.DataFrame([test], columns = ['text'])
#     X_test_text = text_preprocess(X_test)
#     X_test_textp = pipeline_text.transform(X_test_text).toarray()
#     if pipeline_model.predict(X_test_textp)[0] == 0:
#         message = 'Not a disaster.'
#     else: 
#         message = 'A disaster!'
#     # return test
#     return "<xmp>" + message + " </xmp> "


"""
To run, go to anaconda cmd, go to the directory of this file, and execute
(with FLASK_ENV = "development", webpage updates without having to restart server)

$env:FLASK_APP="1_hello_world.py"
$env:FLASK_ENV = "development"   
flask run

 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
alias:
locahost:5000/


Alternatively, to start server by running this script, use

# if __name__ == '__main__':
#     app.run(debug = True)
"""












