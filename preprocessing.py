# -*- coding: utf-8 -*-
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder

def preprocess(filepath): 
    # Read file and encode "sentiment"
    df = pd.read_csv(filepath, usecols = ['sentiment', 'text'])
    lb = LabelEncoder()
    lb.fit(df['sentiment'])
    # print(list(lb.classes_))
    df['sentiment']= lb.fit_transform(df['sentiment'])
    
    # Make "text" more clean
    df['text'] = df['text'].apply(lambda x : clean_text(x))
    
    # Save the dict that map text and encoding
    map_en = dict(enumerate(list(lb.classes_)))
    return df, map_en

def clean_text(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text) 
    return text

