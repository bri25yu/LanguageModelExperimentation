from scripts.create_graph import *


"""
This is currently in progress, with missing results for 1B and 3B models. 

Flores200 graphs are found at
https://github.com/bri25yu/LanguageModelExperimentation/tree/d29d25047e86a2ef73dfbcd0f2ef8b0757a6df18
"""
plot_comparative_experiment(
    ["FloresBaseline600MExperiment", "FloresScaffoldInputMix3600MExperiment", "FloresPackedMix600MExperiment"],
    ["mT5 600M baseline", "mT5 600M English scaffold", "mT5 600M packed in context"],
    "",
    "eval/chrf++",
    "Validation set chrF++ score",
    "flores200_600m",
)
