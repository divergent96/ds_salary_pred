# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:13:51 2021

@author: Susheel
"""

#############################################################################################
# We will use Flask to host model locally and test it out
from flask import Flask
app = Flask(__name__) # create a Flask app

@app.route("/")
def hello():
    return "Hello World!"

if __name__=='__main__':
    app.run(port=3000, debug=True)

#############################################################################################


#############################################################################################