from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd
import numpy as np
import string
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

NLTK_DATA = '../assets/data/nltk_data'

nltk.download(['stopwords', 'punkt', 'wordnet'], download_dir=NLTK_DATA)

nltk.data.path.append(NLTK_DATA)
