from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

import pandas as pd

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

# create pipeline for normalizing and fitting
model = Pipeline([('normalizer', Normalizer()), ('classifier', SVC())])

# create parameters for grid search
params = {'classifier__C':[0.001, 0.01, 0.1, 0.5, 1]}

# create grid search object
# uses classifier.set_params method
grid_search = GridSearchCV(model, params)

# split data to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# fit model and choose best classifier
grid_search.fit(X_train, y_train)
model = grid_search.estimator

# fit best model
model.fit(X_train, y_train)

# evaluate accuracy on train data
# you can also use model.score(X_train, y_train)
train_accuracy = accuracy_score(model.predict(X_train), y_train)

# evaluate accuracy on test data
test_accuracy = accuracy_score(model.predict(X_test), y_test)

print(f'Train accuracy is {train_accuracy}.')
print(f'Test accuracy is {test_accuracy}.')

# save model to disk
joblib.dump(model, 'model.pkl')
