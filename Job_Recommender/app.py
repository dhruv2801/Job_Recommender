from __future__ import division, print_function
import os
import pandas as pd
import json
import glob
import re
import numpy as np
import operator
from pyresparser import ResumeParser
from fuzzywuzzy import fuzz

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


wordnet_lemmatizer = WordNetLemmatizer()
porter=PorterStemmer()

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
application = app = Flask(__name__)

# Load fetched jobs
dataframe = pd.read_csv('models/da_ont.csv')

print('Dataframe loaded. Check http://127.0.0.1:5000/')

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    stop_words = set(stopwords.words('english'))
    for word in token_words:
        if word not in stop_words:
          #wrd = porter.stem(word)  
          stem_sentence.append(wordnet_lemmatizer.lemmatize(word)) 
          stem_sentence.append(" ")
    return "".join(stem_sentence)

def model_predict(res_path, data):    
    resume = []
    indx = []
    score = []
    resume = ResumeParser(res_path).get_extracted_data()
    
    skills = resume.get("skills", "")
    skillset = stemSentence(' '.join(skills))
    dct = {i : fuzz.token_set_ratio(skillset.lower(),data['lemtext'][i].lower()) for i in data.index}
    sorted_d = sorted(dct.items(), key=operator.itemgetter(1),reverse=True)

    for i in range(0,15):
      score.append(sorted_d[i][1])
      indx.append(sorted_d[i][0])
	  
    res = dataframe.iloc[indx]
    res.loc[:,('Similarity Score')] = ""
    res.loc[:,('Salary')] = res.loc[:,('Salary')].fillna("NA")
    res = res.reset_index()
    for i in res.index:
      res.loc[i,('Similarity Score')] = score[i]
	  
    res.index = np.arange(1, len(res) + 1)
    result = res.loc[:,['Title','Location','Company','Salary','Similarity Score']]
    return result


@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

          # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, dataframe)	

        return render_template('view.html',  tables=[preds.to_html(classes='data', header="true")])
		
    elif request.method == 'GET':
       print("No Post Back Call")
		
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


