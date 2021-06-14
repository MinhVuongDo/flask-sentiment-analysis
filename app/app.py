import numpy as np
import warnings
from utils import predict, load_model
from flask import Flask,jsonify,request,render_template,redirect, url_for
from model import SentimentAnalysisModel
warnings.filterwarnings("ignore")
import tensorflow as tf
app = Flask(__name__)
global model 


@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
  if request.method == 'GET':
    return render_template('login.html')
  if request.method == 'POST':
      user = request.form
     
      text = user['name']
      print(text)
      pred = predict(str(text), model)
      # return redirect(url_for())
      print(pred)
      return str(pred)
  
if __name__ == '__main__':
    
    model = load_model(checkpoint_dir = './model')

    app.run(debug = True, port = 5000)