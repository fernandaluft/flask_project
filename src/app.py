import os
os.environ['FLASK_ENV'] = 'production'

import flask
from flask import Flask, render_template, request, redirect, url_for
from pickle import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

vector = load(open("vector_tfidf.sav", "rb"))
model = load(open("svm_sentiment_analysis.sav", "rb"))

def predict_sentiment(str):
    sentence = [str.strip().lower().replace('\t', ' ').replace('\n', ' ').replace('.', '')]
    sentence_vector = vector.transform(sentence).toarray()
    prediction = model.predict(sentence_vector)
    if prediction == 1:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/', methods=['GET', 'POST'])
def rootpage():
    review = ''
    result = ''

    if request.method == 'POST' and 'review' in request.form:
        review = request.form.get('review')
        if review is None:
            result = 'Enter a review'
        else:
            result = predict_sentiment(review)

    return render_template('index.html', result=result)
