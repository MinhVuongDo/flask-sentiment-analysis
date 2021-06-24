import sqlite3
import os
# os.remove('database.db')
conn = sqlite3.connect('database.db')
print("Opened database successfully")

# conn.execute('''CREATE TABLE reviews_table (review_id INT IDENTITY(1,1) PRIMARY KEY, 
# 	name_res TEXT,name_food TEXT,review TEXT, sentiment INTEGER);''')

print("Table created successfully")
cur = conn.cursor()
cur.execute('''INSERT INTO reviews_table(name_res, name_food, review, sentiment)
VALUES ('AK_RESTAUTRANT', 'pho', 
			'rat ngon', 1) ;''')
print('insert successfully')
res = cur.execute('''SELECT * FROM reviews_table;''')
review = res.fetchall()
print(review)
conn.commit()
conn.close()