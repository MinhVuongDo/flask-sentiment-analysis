import numpy as np
import warnings
import sqlite3 as sql
import tensorflow as tf
from app.utils import _predict,load_model
from flask import Flask,jsonify,request,render_template,redirect, url_for
warnings.filterwarnings("ignore")


app = Flask(__name__)

global model
model = load_model(checkpoint_dir = './app/model/')
if model :
  print('model loaded')


@app.route('/predict',methods = ['POST', 'GET'])
def predict():
  if request.method == 'GET':
    con = sql.connect("database.db")
    cur = con.cursor()
    res = cur.execute('''SELECT * FROM reviews_table;''')
    reviews = res.fetchall()
    return render_template('predict.html',reviews=reviews)
  if request.method == 'POST':
    name_res = request.form['name_res']
    name_food = request.form['name_food']
    review = request.form['review']
    pred = _predict(str(review), model)
    with sql.connect("database.db") as con:
      cur = con.cursor()
      cur.execute('''INSERT INTO reviews_table (name_res,name_food,review,sentiment) 
          VALUES (?,?,?,?)''',(name_res,name_food,review,int(pred)) )
      con.commit()
      
    con.close()   
    return redirect(url_for('predict'))
  
