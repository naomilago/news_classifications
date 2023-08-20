from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from config.constants import *
from datetime import datetime
from config.config import *
from loguru import logger
import pandas as pd
import numpy as np
import demoji
import joblib
import string
import json
import nltk

NLTK_DATA = '../assets/data/nltk_data'

if not os.path.exists('../assets/data/nltk_data/corpora') or not os.path.exists('../assets/data/nltk_data/tokenizers'):
    import ssl

    try:
        # noinspection PyProtectedMember
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download(['stopwords', 'punkt', 'wordnet'], download_dir=NLTK_DATA)

nltk.data.path.append(NLTK_DATA)

STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)

class EmptyDataframe(Exception):
    pass

class MissingColumn(Exception):
    pass

def read_data(filename: str, chunk_size: int = 1000):
    with open(filename, 'rb') as f:
        data = pd.read_pickle(f)
        data = data.sample(chunk_size, random_state=20)
        if len(data) == 0:
            raise EmptyDataframe('The dataframe returned with 0 values.')
        yield data

def tokenizer(text: str, stopwords: set = STOPWORDS, punctuations: set = PUNCTUATIONS) -> tuple[list, str]:
    emojis = demoji.findall(text)

    new_text = ''
    for character in text:
        if character in emojis.keys():
            new_text += str.join('_', str.split(emojis[character])) + ' '
        else:
            new_text += character

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokenize(new_text)]
    tokens = [str.lower(token) for token in lemmas if token not in stopwords and token not in punctuations]

    refined_text = str.join(' ', tokens)

    if len(tokens) == 0:
        return ['NULL'], 'NULL'
    else:
        return tokens, refined_text