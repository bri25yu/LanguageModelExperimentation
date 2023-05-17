import pandas as pd
import os 
from IPython.display import display


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATASETS = ['flores200_devtest_mt5-600m-flores200-baseline', 'flores200_devtest_mt5-600m-flores200-scaffold', 'flores200_devtest_mt5-600m-flores200-packed']

loaded = []
names = ['baseline', 'scaffold', 'packed']

for INPUT_DATASET, name in zip(INPUT_DATASETS, names):
    p = pd.read_csv(os.path.join(BASE_DIR, INPUT_DATASET, 'avg_stats.csv'))
    p.rename(columns={'chrF++': 'chrF++_{}'.format(name)}, inplace=True)
    loaded.append(p)

# Merge all on the label column
merged = loaded[0]
for i in range(1, len(loaded)):
    merged = merged.merge(loaded[i], on='label')

# display and save
display(merged)
pd.DataFrame.to_csv(merged, os.path.join(BASE_DIR, 'agg_avg_table.csv'), index=False)
