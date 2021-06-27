from flask_sqlalchemy import SQLAlchemy

db =SQLAlchemy()

class ReviewsModel(db.Model):
    __tablename__ = "reviews"

    id = db.Column(db.Integer, primary_key=True)
    name_res = db.Column(db.String())
    name_food = db.Column(db.String())
    review = db.Column(db.String())
    sentiment = db.Column(db.Integer())

    def __init__(self,name_res,name_food,review,sentiment):
        self.name_res = name_res
        self.name_food = name_food
        self.review = review
        self.sentiment = sentiment

    