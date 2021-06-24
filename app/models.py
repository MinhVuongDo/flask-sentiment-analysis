
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name_restaurant = db.Column(db.String(255))
    name_food = db.Column(db.String(255))
    review = db.Column(db.String())
    sentiment = db.Column(db.Integer)

    def __init__(self,name_restaurant, name_food, review,sentiment):
        self.name_restaurant = name_restaurant
        self.name_food = name_food
        self.review = review
        self.sentiment = category
        

    def __repr__(self):
        return '<Review %d>' % self.id