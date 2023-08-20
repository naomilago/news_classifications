from config.utils import *
import os

DATA_INPUTS = dict({
  'main_dataset': read_data(PROJECT_PATHS['main_dataset'], chunk_size=10000)
})

DATA_PROCESS = dict({
  'label_encoder': os.path.join(PROJECT_PATHS['artifacts'], 'label_encoder.pkl')
})

DATA_OUTPUTS = dict({

})

EXPERIMENTS_CONSTANTS = dict({
    "news_classifier": {
        "description": "Records metrics and params from train/test results",
        "key_var": "metrics_recording",
        "model_name": "Newery",
        "path": os.path.join(
            PROJECT_PATHS["experiment"], "topic_modeling_v2"
        ),
    }
})

SOLUTIONS_CONSTANTS = dict({
  'today_date': datetime.today().strftime('%Y-%m-%d'),
})