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
import string
import json
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

NLTK_DATA = '../assets/data/nltk_data'

if not os.path.exists('../assets/data/nltk_data/corpora') or not os.path.exists('../assets/data/nltk_data/tokenizers'):
    nltk.download(['stopwords', 'punkt', 'wordnet'], download_dir=NLTK_DATA)

nltk.data.path.append(NLTK_DATA)

STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)
