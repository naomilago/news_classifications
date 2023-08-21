import os

ENV = os.getenv('ENV', 'dev')

SOLUTION = 'news_classification'
AUTHOR = 'Naomi Lago'
VERSION = 1.0

EXPERIMENTS = '../assets/experiments/'
ARTIFACTS = '../assets/artifacts/'
DATA = '../assets/data/'

PROJECT_PATHS = dict({
  'artifacts': os.path.join(ARTIFACTS),
  'data_folder': os.path.join(DATA, ''),
  'experiment': os.path.join(EXPERIMENTS, SOLUTION),
  'main_dataset': os.path.join(DATA, 'news_dataset_sampled.pkl')
})