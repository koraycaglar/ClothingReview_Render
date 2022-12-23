from flask import Flask,render_template,url_for,request


import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import joblib

import warnings
warnings.filterwarnings("ignore")

"""
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stopwords = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()
"""


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	fulldata = pd.read_csv('data.csv')
	data = fulldata[['Title', 'Review Text', 'Recommended IND']]
	data['Title'] = np.where(data['Title'].isnull(), '', data['Title'])
	data['Review Text'] = np.where(data['Review Text'].isnull(), '', data['Review Text'])
	data['fullreview'] = data['Title'].map(str) +'. ' + data['Review Text'].map(str)
	data = data[['fullreview', 'Recommended IND']]

	X = data['fullreview']
	y = data['Recommended IND']
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


	#clf = MultinomialNB(alpha=1.2)
	#clf.fit(X_train,y_train)

	clf = joblib.load(open('NB_model.pkl', 'rb'))
		

	if request.method == 'POST':
		message = request.form['message']
		inpt = [message]
		vect = cv.transform(inpt).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)