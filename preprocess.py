
import pandas as pd


# read data file
df = pd.read_csv('./data/waterQuality1.csv')


# Remove rows which is has unwanted values
df = df[df.is_safe != "#NUM!" ]

# convert is_safe to float datatype
df['is_safe'] = df['is_safe'].astype(int)

new = df[['aluminium', 'arsenic', 'cadmium','chloramine', 'chromium','is_safe']].copy()

new.to_csv('water_preprocessed.csv') 
