from config.constants import *
from src.data_prep import data_prep
from config.utils import *

data_generator = read_data(PROJECT_PATHS['main_dataset'], chunk_size=69347)

data_chunks = list([])
for chunk in data_generator:
    data_chunks.append(chunk)

if data_chunks:
  data = pd.concat(data_chunks, ignore_index=True)
else:
  logger.error('The data is empty')

def model_train(df: pd.DataFrame, text: str = 'headline', target: str = 'category', infer: bool = True, verbose: bool = True):
  logger.info(f'Starting to train on a set of {len(df)} entries...')

  logger.info('Preparing the data and splitting into train and test...')
  data_prep(df=df, text=text, target=target, infer=False, verbose=False)

  train = pd.read_pickle(os.path.join(DATA_PROCESS['data_prep'], 'train.pkl'))
  test = pd.read_pickle(os.path.join(DATA_PROCESS['data_prep'], 'test.pkl'))

  logger.info('Starting the model...')
  model = ComplementNB()

  X_train: np.ndarray[int] = train['scaled_vector'].tolist()
  y_train: np.ndarray[int] = train[str.join('', 'le_' + target)].tolist()

  X_test: np.ndarray[int] = test['scaled_vector'].tolist()
  y_test: np.ndarray[int] = test[str.join('', 'le_' + target)].tolist()

  logger.info('Training the model...')
  model.fit(X_train, y_train)

  logger.info('Making predictions...')
  predictions = model.predict(X_test)

  precision = precision_score(y_true=y_test, y_pred=predictions, average='weighted', zero_division=np.nan)
  recall = recall_score(y_true=y_test, y_pred=predictions, average='weighted')
  accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
  f1 = f1_score(y_true=y_test, y_pred=predictions, average='weighted')

  print(f'Precision: {precision} | Recall: {recall} | Accuracy: {accuracy} | F1: {f1} ')

if __name__ == '__main__':
    model_train(df=data, verbose=True)