import sweetviz as sv
import pandas as pd
df = pd.read_csv('drowsiness_dataset.csv')
my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"