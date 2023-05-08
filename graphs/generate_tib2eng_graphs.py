from scripts.create_graph import *


"""
NLLB baseline, mT5 baseline, and mT5 completing an input graphs are found at
https://github.com/bri25yu/LanguageModelExperimentation/tree/07a9b30e05edaf419fb16ced820e04c27c2bd596
"""
plot_comparative_experiment(
    ["TranslationNLLB600MExperiment", "TranslationMT5600MExperiment", "TranslationIncomplete4Experiment"],
    ["NLLB 600M baseline", "mT5 600M baseline", "mT5 600M completing an input"],
    "Classical Tibetan to English performance comparing NLLB and mT5 600M experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_600m",
)

plot_comparative_experiment(
    ["TranslationNLLB1BExperiment", "TranslationMT51BExperiment", "TranslationIncompleteMT51BExperiment"],
    ["NLLB 1B baseline", "mT5 1B baseline", "mT5 1B completing an input"],
    "Classical Tibetan to English performance comparing NLLB and mT5 1B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_1b",
)

plot_comparative_experiment(
    ["TranslationNLLB3BExperiment", "TranslationMT53BExperiment", "TranslationIncompleteMT53BExperiment"],
    ["NLLB 3B baseline", "mT5 3B baseline", "mT5 3B completing an input"],
    "Classical Tibetan to English performance comparing NLLB and mT5 3B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_3b",
)

plot_comparative_experiment(
    ["TranslationMT5600MExperiment", "TranslationIncomplete4Experiment"],
    ["mT5 600M baseline", "mT5 600M completing an input"],
    "Classical Tibetan to English performance for mT5 600M experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_600m",
)

plot_comparative_experiment(
    ["TranslationMT51BExperiment", "TranslationIncompleteMT51BExperiment"],
    ["mT5 1B baseline", "mT5 1B completing an input"],
    "Classical Tibetan to English performance for mT5 1B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_1b",
)

plot_comparative_experiment(
    ["TranslationMT53BExperiment", "TranslationIncompleteMT53BExperiment"],
    ["mT5 3B baseline", "mT5 3B completing an input"],
    "Classical Tibetan to English performance for mT5 3B experimental configurations",
    "eval/bleu_score",
    "Validation set BLEU score",
    "tib2eng_mt5_3b",
)
