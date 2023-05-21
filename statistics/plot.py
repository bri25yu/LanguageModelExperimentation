import math
import matplotlib.pyplot as plt
import os 
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DATASETS = [] 
PARAM_COUNT = '600m'
loaded = []
names = ['baseline', 'scaffold', 'packed']
for name in names:
    INPUT_DATASETS.append('flores200_devtest_mt5-{}-flores200-{}'.format(PARAM_COUNT, name))

PLOT_TYPE = 'lang_to_xx'

fig, ax = plt.subplots()

for INPUT_DATASET in INPUT_DATASETS:
    p = pd.read_csv(os.path.join(BASE_DIR, INPUT_DATASET, f'{PLOT_TYPE}_mt5_pretrain.csv'))

    ax.scatter(x=p['size'].astype(int).apply(lambda x: math.log10(x)), y=p['chrF++'])
    ax.set_xlabel("Pretraining data size (log_10)")
    ax.set_ylabel("chrF++")
    ax.set_title(f"mT5 {PARAM_COUNT} Pretraining data size vs post-finetuning chrF++ score, {PLOT_TYPE}")

ax.legend(names)
plt.show()
