from config.utils import *
import os

DATA_INPUTS = dict({
  'main_dataset': pd.read_pickle(PROJECT_PATHS['main_dataset']).sample(10000, random_state=20)
})

DATA_PROCESS = dict({
  
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