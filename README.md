# PulsarPrediction

Task is to predict whether a star is pulsar. Download dataset from [Kaggle](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star).
All files:
- `pulsar_stars.csv`
- `model.py`
- `profiling.py`
- `predict.py`

We need to overview data, so we use `pandas-profiling` to create profiling report in `.html` file.

## profiling.py
```python
import pandas_profiling as profiling
import pandas as pd

# load data from .csv file to pandas.DataFrame object
df = pd.read_csv('pulsar_stars.csv')

# create report for DataFrame
report = profiling.ProfileReport(df)

# save report as HTML
report.to_file('profile_report.html')
```
We use `pandas.read_csv` for reading data from `.csv` file. More detailed information about this function [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

Then we create profile report with `pandas_profiling.ProfileReport` class. Check [repository of this package on GitHub](https://github.com/pandas-profiling/pandas-profiling).

Using `ProfileReport.to_file` method we save report to `.html` file. Open it in your browser.

Run this file using command `python profiling.py` to create `profile_report.html`.

How you can see, all columns of this frame are numeric, so we don't need to preprocess missing values. Also, we got 4 warning about correlations in this dataset. Since we use SVM (Support Vector Machine), not a Bayes Classifier, we don't need to delete columns with high correlation.

## model.py
Now we are going to create machine learning model. 

Firstly, we should import all functions and classes:
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

import pandas as pd
```

Then we need to read data from `pulsar_stars.csv` and split data to samples and target.
```python
# load data from .csv file to pandas.DataFrame object
df = pd.read_csv('pulsar_stars.csv')

# get target
y = df['target_class']

# delete target column
df = df.drop(['target_class'], axis=1)

# get features
X = df.values

# we already don't need this object
del df
```

#### What about normalizing?
Normalizing is the process of bringing numerical data to a single interval (like (0, 1) or (-1, 1)). 

Why should we do this? 

Image situatuon: we have two features - the first is equal to 5 and the second is equal to 100. At the first iteration of fitting weights of this features are approximately equal.
In this case, the second feature has is more important for classification than the first, what is the wrong assumption.

#### Model's fitting

We use `sklearn.preprocessing.Normalizer` for normalization. This is transformer, i.e. class, that transforms data.
Support Vector Machine (SVM) algorithm was realized in `sklearn.svm`. Use `SVC.fit` method for model's fitting.

#### Pipelines
The good idea is to use `sklearn.pipeline.Pipeline` for automatic normalization and classification. Import this class and set the stages of pre-processing and classification.
```python
# create pipeline for normalizing and fitting
model = Pipeline([('normalizer', Normalizer()), ('classifier', SVC())])
```

#### Grid search of best hyperparameters
The grid search algorithm for searching best combination of hyperparameter for model is very simple - algorithm fits few models for each combination of hyperparameters.

We need to choose only one parameter - `SVC.C`, parameter for regularization. 
```python
# create parameters for grid search
params = {'classifier__C':[0.001, 0.01, 0.1, 0.5, 1]}

# create grid search object
# uses classifier.set_params method
grid_search = GridSearchCV(model, params)
```
