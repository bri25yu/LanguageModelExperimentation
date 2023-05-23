import pandas as pd
import os 
from IPython.display import display


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_COUNT = '3b'
loaded = []
names = ['baseline', 'scaffold', 'packed']

for name in names:
    p = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'script_analysis', 'script_to_script.csv'))
    # Load script_to_xx and xx_to_script
    script_to_xx = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'script_analysis', 'script_to_xx.csv'))
    xx_to_script = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'script_analysis', 'xx_to_script.csv'))
    # Add row for eng-xx and xx-eng
    p = pd.concat([script_to_xx, p])
    p = pd.concat([xx_to_script, p])
    p['chrF++'] = p['chrF++'].round(1)
    p.rename(columns={'chrF++': 'chrF++_{}'.format(name)}, inplace=True)
    loaded.append(p)

# Merge all on the label column
merged = loaded[0]
for i in range(1, len(loaded)):
    merged = merged.merge(loaded[i], on='label')

# display and save
display(merged)
pd.DataFrame.to_csv(merged, os.path.join(BASE_DIR, PARAM_COUNT, 'agg_scripts_table.csv'), index=False)
