import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from flask import Flask, request, render_template

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

df = pd.read_csv("dataset/train.csv")

# EDA
df.info()

# print(df.describe())

# Splitting Dataset
komentar = df['comment_text']  

label_toxic = df['toxic']
label_severe_toxic = df['severe_toxic']
label_obscene = df['obscene']
label_threat = df['threat']
label_insult = df['insult']
label_identity_hate = df['identity_hate']



vec = TfidfVectorizer()
X = vec.fit_transform(df['comment_text'])

toxic = MultinomialNB()
toxic.fit(X, df['toxic'])

severe_toxic = MultinomialNB()
severe_toxic.fit(X, df['severe_toxic'])

obscene = MultinomialNB()
obscene.fit(X, df['obscene'])

threat = MultinomialNB()
threat.fit(X, df['threat'])

insult = MultinomialNB()
insult.fit(X, df['insult'])

identity_hate = MultinomialNB()
identity_hate.fit(X, df['identity_hate'])


def predict_toxic(text):
    value = vec.transform([clean_text(text)])
    res = toxic.predict_proba(value[0])
    return res[:, 1][0]

def predict_severe_toxic(text):
    value = vec.transform([clean_text(text)])
    res = severe_toxic.predict_proba(value[0])
    return res[:, 1][0]

def predict_obscene(text):
    value = vec.transform([clean_text(text)])
    res = obscene.predict_proba(value[0])
    return res[:, 1][0]

def predict_threat(text):
    value = vec.transform([clean_text(text)])
    res = threat.predict_proba(value[0])
    return res[:, 1][0]

def predict_insult(text):
    value = vec.transform([clean_text(text)])
    res = insult.predict_proba(value[0])
    return res[:, 1][0]

def predict_identity_hate(text):
    value = vec.transform([clean_text(text)])
    res = identity_hate.predict_proba(value[0])
    return res[:, 1][0]

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json['text']
    return {
        "text": text,
        "result": {
            "toxic": predict_toxic(text),
            "severe_toxic": predict_severe_toxic(text),
            "obscene": predict_obscene(text),
            "threat": predict_threat(text),
            "insult": predict_insult(text),
            "identity_hate": predict_identity_hate(text),
        }
    }

