import math
import matplotlib.pyplot as plt
import os 
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARAM_COUNT = '1b'
loaded = []
names = ['baseline', 'scaffold', 'packed']
PLOT_TYPE = 'script_to_script'

Xlabels = ['Arab', 'Cyrl', 'Latn', 'other']
Xticks = np.arange(len(Xlabels))
Xoffsets = np.array([-.2, 0, .2])

fig, ax = plt.subplots()

for i, name in enumerate(names):
    p = pd.read_csv(os.path.join(BASE_DIR, PARAM_COUNT, name, 'script_analysis', f'{PLOT_TYPE}.csv'))

    if PLOT_TYPE == 'script_to_script':
        p = p[p['label'].apply(lambda x: x.split('-')[1] == x.split('-')[0])]

    ax.bar(x=Xticks + Xoffsets[i], height=p['chrF++'], width=.2)

plt.xticks(Xticks, Xlabels)
ax.set_xlabel("Label")
ax.set_ylabel("chrF++")
ax.set_title(f"mT5 {PARAM_COUNT} {PLOT_TYPE}")

ax.legend(names)
plt.show()
