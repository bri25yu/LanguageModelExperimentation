from graphs.create_graph import *


plot_comparative_series(
    batch_graph_kwargs=[
        {
            "experiment_names": ["FloresBaseline600MExperiment", "FloresScaffoldInputMix3600MExperiment", "FloresPacked600MExperiment"],
            "legend_labels": ["mT5 600M baseline", "mT5 600M English scaffold", "mT5 600M packed in context"],
            "title": "mT5 600M",
        },
        {
            "experiment_names": ["FloresBaseline1BExperiment", "FloresScaffoldInputMix31BExperiment", "FloresPacked1BExperiment"],
            "legend_labels": ["mT5 1B baseline", "mT5 1B English scaffold", "mT5 1B packed in context"],
            "title": "mT5 1B",
        },
        {
            "experiment_names": ["FloresBaseline3BExperiment", "FloresScaffold3BExperiment", "FloresPacked3BExperiment"],
            "legend_labels": ["mT5 3B baseline", "mT5 3B English scaffold", "mT5 3B packed in context"],
            "title": "mT5 3B",
        },
    ],
    title="Flores200 performance for mT5 experimental configurations",
    property_name="eval/chrf++",
    y_label="Validation set chrF++ score",
    save_name="flores200",
    y_lim=[0, 27],
)
