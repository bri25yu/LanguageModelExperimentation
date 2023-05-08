from scripts.create_graph import *


"""
NLLB baseline, mT5 baseline, and mT5 completing an input graphs are found at
https://github.com/bri25yu/LanguageModelExperimentation/tree/07a9b30e05edaf419fb16ced820e04c27c2bd596
"""
plot_comparative_experiment(
    ["TranslationMT5600MExperiment", "TranslationIncomplete4Experiment", "TranslationNLLB600MExperiment"],
    ["mT5 600M baseline", "mT5 600M completing an input", "NLLB 600M baseline"],
    "Comparing NLLB and mT5 600M\non Classical Tibetan to English",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_600m",
)

plot_comparative_experiment(
    ["TranslationMT51BExperiment", "TranslationIncompleteMT51BExperiment", "TranslationNLLB1BExperiment"],
    ["mT5 1B baseline", "mT5 1B completing an input", "NLLB 1B baseline"],
    "Comparing NLLB and mT5 1B\non Classical Tibetan to English",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_1b",
)

plot_comparative_experiment(
    ["TranslationMT53BExperiment", "TranslationIncompleteMT53BExperiment", "TranslationNLLB3BExperiment"],
    ["mT5 3B baseline", "mT5 3B completing an input", "NLLB 3B baseline"],
    "Comparing NLLB and mT5 3B\non Classical Tibetan to English",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_3b",
)

plot_comparative_experiment(
    ["TranslationMT5600MExperiment", "TranslationIncomplete4Experiment"],
    ["mT5 600M baseline", "mT5 600M completing an input"],
    "Classical Tibetan to English performance\nfor mT5 600M experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_600m",
)

plot_comparative_experiment(
    ["TranslationMT51BExperiment", "TranslationIncompleteMT51BExperiment"],
    ["mT5 1B baseline", "mT5 1B completing an input"],
    "Classical Tibetan to English performance\nfor mT5 1B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_1b",
)

plot_comparative_experiment(
    ["TranslationMT53BExperiment", "TranslationIncompleteMT53BExperiment"],
    ["mT5 3B baseline", "mT5 3B completing an input"],
    "Classical Tibetan to English performance\nfor mT5 3B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_3b",
)
