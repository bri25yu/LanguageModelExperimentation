import pandas as pd
import os 
from IPython.display import display


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATASETS = [] 
PARAM_COUNT = '600m'
loaded = []
names = ['baseline', 'scaffold', 'packed']
for name in names:
    INPUT_DATASETS.append('flores200_devtest_mt5-{}-flores200-{}'.format(PARAM_COUNT, name))

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
pd.DataFrame.to_csv(merged, os.path.join(BASE_DIR, f'agg_avg_table_{PARAM_COUNT}.csv'), index=False)
