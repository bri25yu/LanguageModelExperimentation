import math
import matplotlib.pyplot as plt
import os 
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DATASET = 'flores200_devtest_mt5-600m-flores200-scaffold'
p = pd.read_csv(os.path.join(BASE_DIR, INPUT_DATASET, 'xx_to_lang_mt5_pretrain.csv'))

fig, ax = plt.subplots()
ax.scatter(x=p['size'].astype(int).apply(lambda x: math.log10(x)), y=p['chrF++'])
ax.set_xlabel("Pretraining data size (log_10)")
ax.set_ylabel("chrF++")
ax.set_title("mT5 Pretraining data size vs post-finetuning chrF++ score, lang-xx")
plt.show()