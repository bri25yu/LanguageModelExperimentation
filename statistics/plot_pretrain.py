import math
import matplotlib.pyplot as plt
import os 
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARAM_COUNT = '600m'
loaded = []
names = ['baseline', 'scaffold', 'packed']
PLOT_TYPE = 'lang_to_xx'

fig, ax = plt.subplots()

for name in names:
    p = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'pretraining', f'{PLOT_TYPE}_mt5_pretrain.csv'))

    ax.scatter(x=p['size'].astype(int).apply(lambda x: math.log10(x)), y=p['chrF++'])
    ax.set_xlabel("Pretraining data size (log_10)")
    ax.set_ylabel("chrF++")
    ax.set_title(f"mT5 {PARAM_COUNT} Pretraining data size vs post-finetuning chrF++ score, {PLOT_TYPE}")

ax.legend(names)
plt.show()
