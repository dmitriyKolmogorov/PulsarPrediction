# PulsarPrediction

Task is to predict whether a star is pulsar. Download dataset from [Kaggle](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star).

We need to overview data, so we use `pandas-profiling` to create profiling report in `.html` file.

# profiling.py
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
