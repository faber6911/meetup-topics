#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
try: 
    import emoji
except ImportError:
    get_ipython().system('pip install emoji --user')
    import emoji
    
#try:
#    from bs4 import BeautifulSoup 
#except:
#    !pip install bs4 --user
#    from bs4 import BeautifulSoup
from tqdm.autonotebook import  tqdm
tqdm.pandas()

try:
    import nltk
    #raise Exception
except:
    get_ipython().system('pip install nltk --user')
    import nltk
    nltk.download('punkt')
    
#try:
#    import textblob
#except:
#    !pip install textblob --user
#    import textblob
import re

try:
    from polyglot.detect import Detector
except ImportError:
    #!pip install --user pyicu pycld2
    get_ipython().system('pip install polyglot --user')

    from polyglot.detect import Detector
    
try:
    from googletrans import Translator
except:
    get_ipython().system('pip install googletrans --user')
    from googletrans import Translator


# In[ ]:

#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

import tensorflow
print("tf", tensorflow.__version__)


# In[2]:


from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

import gensim
from gensim.models.doc2vec import TaggedDocument

from tensorflow.keras.models import load_model

set(stopwords.words('english'))
app = Flask(__name__)


df = pd.read_pickle("/home/dario/tm-project/data/df_preprocessed_eng_remap.pckle")

doc2vec_model = gensim.models.Doc2Vec.load("model_doc2vec")

#model = load_model("NN_model.h5")
model = load_model("model_new.h5")



def vec_for_learning(model, tagged_docs):
    #sents = tagged_docs.values
    regressors = [model.infer_vector(doc.words, steps=20) for doc in tqdm(tagged_docs.values)]
    return  regressors

def fake_tagged_doc(desc):
    arr=np.asarray(desc)
    arr=pd.Series(arr)

    test_tagged = arr.apply( lambda r: TaggedDocument(words=str(r).split(" "), tags=["NaN"]))#, axis=1)
    return test_tagged



import emoji

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text

def detect_lang(text):
    try:
        lang = Detector(text, quiet=True)
        if lang.reliable:
            return lang.language.name  if lang.language.confidence > 50 else "low_conf"
        else: 
            return "not_reliable"
    except Exception as e: 
        return "error"


# In[8]:


europ_languages = ["english", 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish']
   
stopwords = set(nltk.corpus.stopwords.words(europ_languages))    

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

#text = "Another way of achieving this task"
#tokens = word_tokenize(text)
lemmatizer = WordNetLemmatizer()

badwords=[
    "www", "http", "https", "th", "pm", "ticket", "org", "event", "link", "registr", "hi", "oil", "en", "lo", "ca", 
    "month", "monday", "tuesday", "wednesday", "thursday","friday", "saturday", "sunday",
    "meetup","meetups", "meet","area","happen", "event", "group", "regist", "pleas", "please", "join", "rsvp", "member", "mask", 
    "venu", "free", "comment", "thank", "attend", "eventbrit", "mr", "st", "rd", "hour", "mask", "locat", "everyone", "everyon", "contact", "anyone", "great",
    "new", "time", "stand", "host", "check", "line", "com", "fee", "cost", "people", "day", "new", "know", "inform", "email", "bring","welcome", "welcom",
    "boston", "like", "la", "en", "los", "come", "let", "facebook", "available",  "help", "look", "register", "sign","registration",  ]
len(badwords), len(set(badwords)) #woops


# In[9]:


def preproc(raw_text, badwords = badwords, lemmatizer = lemmatizer, stopwords = stopwords, tag_map = tag_map):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", raw_text)
    text = give_emoji_free_text(text)
    lang = detect_lang(text)
    
    
    if lang != "English":
        langs=Detector(text).languages
        lang=langs[0].name.lower()        
        translator= Translator()#from_lang=langs[0].code, to_lang="en")
        text = translator.translate(text).text
    
    text = text.replace("'s ", " ")
    text = re.sub("[^a-zA-Z]", " ", text).lower().split( ) #.replace("|","").replace("!","").replace("?","")
    text = [token for token in text if all([token not in stopwords, token not in badwords])] 
    text = [lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(text)]
    text = [elem for elem in text if elem not in badwords] #badwords after stemming
    return (" ".join(text))


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

lb = LabelEncoder().fit(df.remap_category)


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post(preproc = preproc, fake_tagged_doc=fake_tagged_doc, vec_for_learning = vec_for_learning, model = model, doc2vec_model = doc2vec_model):
    text1 = request.form['text1'].lower()

    clean_test=preproc(text1)
    tag_doc=fake_tagged_doc(clean_test)
    desc_vec=vec_for_learning(doc2vec_model, tag_doc)
    predicted_category=model.predict_classes(np.array(desc_vec))
    proba = np.max(model.predict(np.array(desc_vec)))
    
    return render_template('form.html', final=round(proba, 2), text1=text1, category = lb.inverse_transform(predicted_category))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)





