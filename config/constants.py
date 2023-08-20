from datetime import datetime
from config.config import *
import os

DATA_INPUTS = dict({
  
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