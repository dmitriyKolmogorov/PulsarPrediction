import pandas_profiling as profiling
import pandas as pd

# load data from .csv file to pandas.DataFrame object
df = pd.read_csv('pulsar_stars.csv')

# create report for DataFrame
report = profiling.ProfileReport(df)

# save report as HTML
report.to_file('profile_report.html')