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
            logger.info('This code is still in development...')
            le = joblib.load(DATA_PROCESS['label_encoder'])
            df[str.join('', 'le_' + target)] = le.transform(df[target].tolist())
            print(df.head())
        else:
            pass
    else:
        pass


if __name__ == '__main__':
    data_prep(df=data, infer=False, verbose=True)