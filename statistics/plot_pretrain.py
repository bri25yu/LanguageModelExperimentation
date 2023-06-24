from os.path import abspath, dirname, join

from matplotlib.pyplot import subplots
from matplotlib import rcParams
rcParams.update({"font.size": 24})

from pandas import read_csv


BASE_DIR = dirname(abspath(__file__))
save_path = join(BASE_DIR, "flores_pretrain_size.pdf")


plot_types = ["lang_to_xx", "xx_to_lang"]
plot_type_titles = ["from", "into"]
param_counts = ["600m", "1b", "3b"]
names = ["baseline", "scaffold", "packed"]
labels = ["baseline", "ParSE", "MiPS"]


rows, cols = 2, 3
fig, axs = subplots(rows, cols, figsize=(10 * cols, 8 * rows), sharex=True, sharey=True)

for plot_type, plot_type_title, plot_type_axs in zip(plot_types, plot_type_titles, axs):
    for param_count, param_count_ax in zip(param_counts, plot_type_axs):
        for name, label in zip(names, labels):
            df = read_csv(join(BASE_DIR, param_count, name, "pretraining", f"{plot_type}_mt5_pretrain.csv"))
            param_count_ax.scatter(df["size"], df["chrF++"], label=label)

        param_count_ax.set_xscale("log")
        param_count_ax.set_title(f"mT5 {param_count}, translation {plot_type_title}")
        param_count_ax.legend()

fig.suptitle("Pretrain dataset size vs chrF++ score")
fig.supxlabel("Pretrain dataset # examples")
fig.supylabel("chrF++ score")

fig.tight_layout()
fig.savefig(save_path, format="pdf")
