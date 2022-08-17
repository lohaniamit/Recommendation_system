from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

model = pickle.load(open('files/sent_xgboost_model.pkl','rb'))
vectorizer=pd.read_pickle('files/tfidf-vectorizer.pkl')
user_final_rating=pickle.load(open('files/user_user_final_rating.pkl','rb'))
data=pd.read_csv("sample30.csv")
cleaned_data=pickle.load(open('files/data_cleaned.pkl','rb'))
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def getRecommendationByUser(user):
    recommedations = []
    return list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

def getSentimentRecommendations(user):
    if (user in user_final_rating.index):
        # get the product recommedation using the trained ML model
        recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        filtered_data = cleaned_data[cleaned_data.id.isin(recommendations)]

        X = vectorizer.transform(
            filtered_data["reviews_text_cleaned"].values.astype(str))
        filtered_data["predicted_sentiment"] = model.predict(X)
        temp = filtered_data[['id', 'predicted_sentiment']]
        temp_grouped = temp.groupby('id', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
            temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(
            temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
        sorted_products = temp_grouped.sort_values(
            'pos_sentiment_percent', ascending=False)[0:5]
        return pd.merge(data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])

    else:
        print(f"User name {user} doesn't exist")
        return None
