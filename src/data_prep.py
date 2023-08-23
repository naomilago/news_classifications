from config.constants import *
from config.utils import *

data_generator = DATA_INPUTS['main_dataset']

data_chunks = list([])
for chunk in data_generator:
    data_chunks.append(chunk)

if data_chunks:
  data = pd.concat(data_chunks, ignore_index=True)
else:
  logger.error('The data is empty')

def data_prep(df: pd.DataFrame, text: str = 'headline', target: str = 'category', infer: bool = True, verbose: bool = True):
    if verbose:
        if not infer:
            logger.info('Preparing data for training...')
            logger.info('Label encoding the targets...')

            le = joblib.load(DATA_PROCESS['label_encoder'])
            df[str.join('', 'le_' + target)] = le.transform(df[target].tolist())

            logger.info('Tokenizing the texts...')
            df['tokens'], df['refined_text'] = df[text].apply(lambda x: tokenizer(x)[0]), df[text].apply(lambda y: tokenizer(y)[-1])

            logger.info('Getting the text vectors...')
            df['vector'] = df['refined_text'].apply(lambda z: get_ids(z))

            logger.info('Scaling the vectors from 0 to 1...')
            df['scaled_vector'] = df.vector.apply(lambda x: vector_scaler(x))

            logger.info('Splitting into train/test...')
            train, test = train_test_split(df, test_size=.2, random_state=20)

            logger.success('Data prepared successfully for training 🎉')
            train.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'train.pkl'))
            test.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'test.pkl'))

            return dict({
                'train': train,
                'test': test
            })
        else:
            logger.info('Preparing data for inference...')
            logger.info('Tokenizing the texts...')
            df['tokens'], df['refined_text'] = df[text].apply(lambda x: tokenizer(x)[0]), df[text].apply(
                lambda y: tokenizer(y)[-1])

            logger.info('Getting the text vectors...')
            df['vector'] = df['refined_text'].apply(lambda z: get_ids(z))

            logger.info('Scaling the vectors from 0 to 1...')
            df['scaled_vector'] = df.vector.apply(lambda x: vector_scaler(x))

            logger.success('Data prepared successfully for inference 🎉')
            df.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'infer.pkl'))

            return df
    else:
        if not infer:
            le = joblib.load(DATA_PROCESS['label_encoder'])
            df[str.join('', 'le_' + target)] = le.transform(df[target].tolist())

            df['tokens'], df['refined_text'] = df[text].apply(lambda x: tokenizer(x)[0]), df[text].apply(lambda y: tokenizer(y)[-1])

            df['vector'] = df['refined_text'].apply(lambda z: get_ids(z))

            df['scaled_vector'] = df.vector.apply(lambda x: vector_scaler(x))

            train, test = train_test_split(df, test_size=.2, random_state=20)

            train.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'train.pkl'))
            test.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'test.pkl'))

            return dict({
                'train': train,
                'test': test
            })
        else:
            df['tokens'], df['refined_text'] = df[text].apply(lambda x: tokenizer(x)[0]), df[text].apply(
                lambda y: tokenizer(y)[-1])

            df['vector'] = df['refined_text'].apply(lambda z: get_ids(z))

            df['scaled_vector'] = df.vector.apply(lambda x: vector_scaler(x))

            df.to_pickle(os.path.join(DATA_PROCESS['data_prep'], 'infer.pkl'))

            return df

if __name__ == '__main__':
    data_prep(df=data.sample(10, random_state=20), infer=False, verbose=True)