import os

ENV = os.getenv('ENV', 'dev')

SOLUTION = 'news_classification'
AUTHOR = 'Naomi Lago'
VERSION = 1.0

EXPERIMENTS = '../assets/experiments/'
DATA = '../assets/data/'

PROJECT_PATHS = dict({
  'experiment': os.path.join(EXPERIMENTS, SOLUTION),
  'main_dataset': os.path.join(DATA, 'news_dataset_sampled.pkl')
})