import pandas as pd
import pandas_profiling

data = pd.read_csv("allsize.csv")
print(data.describe())

# profile = data.profile_report(title='steel')
# profile.to_file(output_file='steel_report.html')