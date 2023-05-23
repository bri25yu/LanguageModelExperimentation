import pandas as pd
import os 
from IPython.display import display


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_COUNT = '600m'
loaded = []
names = ['baseline', 'scaffold', 'packed']

for name in names:
    p = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'avg_stats.csv'))
    p = p[p['label'] != 'xx-yy']
    # Load lang_to_xx and xx_to_lang
    lang_to_xx = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'pretraining', 'lang_to_xx_mt5_pretrain.csv'))
    xx_to_lang = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'pretraining', 'xx_to_lang_mt5_pretrain.csv'))
    # Add row for eng-xx and xx-eng
    p = pd.concat([pd.DataFrame({'label': 'eng-xx', 'chrF++': lang_to_xx[lang_to_xx['label'] == 'eng_Latn-xx']['chrF++']}), p])
    p = pd.concat([pd.DataFrame({'label': 'xx-eng', 'chrF++': xx_to_lang[xx_to_lang['label'] == 'xx-eng_Latn']['chrF++']}), p])
    p['chrF++'] = p['chrF++'].round(1)
    p.rename(columns={'chrF++': 'chrF++_{}'.format(name)}, inplace=True)
    loaded.append(p)

# Merge all on the label column
merged = loaded[0]
for i in range(1, len(loaded)):
    merged = merged.merge(loaded[i], on='label')

# display and save
display(merged)
pd.DataFrame.to_csv(merged, os.path.join(BASE_DIR, PARAM_COUNT, f'agg_avg_table.csv'), index=False)
