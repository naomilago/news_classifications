from config.constants import *
from config.utils import *

data_generator = DATA_INPUTS['main_dataset']

data_chunks = list([])
for chunk in data_generator:
    data_chunks.append(chunk)

data = pd.concat(data_chunks, ignore_index=True)

def data_prep(df: pd.DataFrame, text: str = 'headline', target: str = 'category', infer: bool = True, verbose: bool = True):
    if verbose:
        if not infer:
            logger.info('Preparing data for inference...')
            logger.info('Label encoding the targets...')

            le = joblib.load(DATA_PROCESS['label_encoder'])
            df[str.join('', 'le_' + target)] = le.transform(df[target].tolist())

            logger.info('Tokenizing the texts...')
            df['tokens'], df['refined_text'] = df[text].apply(lambda x: tokenizer(x)[0]), df[text].apply(lambda y: tokenizer(y)[-1])

            print(get_ids('hi catapimbas', max_length=5))

        else:
            pass
    else:
        pass


if __name__ == '__main__':
    data_prep(df=data.sample(10, random_state=20), infer=False, verbose=True)