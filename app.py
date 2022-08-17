from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
import model as sentiment_model

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    print(f'Recommendation for {user}')
    items = sentiment_model.getSentimentRecommendations(user)
    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)
        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="User Name doesn't exists")
    

if __name__ == '__main__':
    app.run()
