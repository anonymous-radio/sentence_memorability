# Sentence memorability

Anonymous NEURIPS submission

## `exp` directory: behavioral experiment
Our experiment uses jsPsych and is hosted on university servers. Note that some of the code has been redacted for anonymity. We are releasing the following:
- `templates/memexpt.html`: the html file with embedded JavaScript code that comprises the bulk of the experiment
- `static/*`: utility functions and scripts used by the experiment
- `static/data/`: experimental materials

## `analysis` directory: code for analysis
- `cv_predictors.py`: fit cross-validated linear models for one or more predictors, predicting empirical sentence memorability 
- `exp_stats.py`: compute statistics for the experimental
- `inter_participant_cv.py`: perform inter-participant correlation analysis
- `performance_metrics.py`: compute aggregated performance metrics: accuracy, hit rate, false alarm rate

## `embeddings` directory: code for handling word and sentence embeddings
- `sentence_embeddings.py`: compute sentence embeddings and cosine distinctiveness scores for sentences
- `embedding_utils.py`: additional utility functions

## `data` directory: data from experiment which we are releasing publicly
- We release `sentence_memorability_data.csv`, a CSV file containing the 1500 critical sentences mentioned in the submission, along with accuracy, hit rate, and false alarm scores for each one. 
- We also include the public word memorability dataset from Tuckute, Mahowald, et al. (2022), which we used to build a pseudo-memorability estimator (predicting memorability from GloVe embeddings). 