from scripts.create_graph import *


plot_comparative_experiment(
    ["FloresBaseline600MExperiment", "FloresScaffoldInputMix3600MExperiment", "FloresPackedMix600MExperiment"],
    ["mT5 600M baseline", "mT5 600M English scaffold", "mT5 600M packed in context"],
    "",
    "eval/chrf++",
    "Validation set chrF++ score",
    "flores200_600m",
)

plot_comparative_experiment(
    ["FloresBaseline1BExperiment", "FloresScaffoldInputMix31BExperiment", "FloresPacked1BExperiment"],
    ["mT5 1B baseline", "mT5 1B English scaffold", "mT5 1B packed in context"],
    "",
    "eval/chrf++",
    "Validation set chrF++ score",
    "flores200_1b",
)

plot_comparative_experiment(
    ["FloresBaseline3BExperiment", "FloresScaffold3BExperiment", "FloresPacked3BExperiment"],
    ["mT5 3B baseline", "mT5 3B English scaffold", "mT5 3B packed in context"],
    "",
    "eval/chrf++",
    "Validation set chrF++ score",
    "flores200_3b",
    y_min=5.0,
)
