import numpy as np
import warnings
# import sqlite3 as sql
import tensorflow as tf
from app.utils import _predict,load_model
from flask import Flask,jsonify,request,render_template,redirect, url_for
warnings.filterwarnings("ignore")
from app.models import db,ReviewsModel

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.before_first_request
def create_table():
    db.create_all()
global model
model = load_model(checkpoint_dir = './app/model/')


@app.route('/predict',methods = ['POST', 'GET'])
def predict():
  if request.method == 'GET':
    reviews = ReviewsModel.query.all()
    
    return render_template('predict.html',reviews=reviews)
  if request.method == 'POST':
    name_res = request.form['name_res']
    name_food = request.form['name_food']
    review = request.form['review']
    pred = _predict(str(review), model)
    _review = ReviewsModel(name_res=name_res, name_food=name_food,
                                review =review,sentiment=int(pred))
    db.session.add(_review)
    db.session.commit()  
    return redirect(url_for('predict'))
  
