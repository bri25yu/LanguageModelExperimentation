from graphs.create_graph import *


plot_comparative_series(
    batch_graph_kwargs=[
        {
            "experiment_names": ["TranslationMT5600MExperiment", "TranslationIncomplete4Experiment", "TranslationNLLB600MExperiment"],
            "legend_labels": ["mT5 600M baseline", "mT5 600M POSE", "NLLB 600M"],
            "title": "mT5 vs NLLB 600M",
        },
        {
            "experiment_names": ["TranslationMT51BExperiment", "TranslationIncompleteMT51BExperiment", "TranslationNLLB1BExperiment"],
            "legend_labels": ["mT5 1B baseline", "mT5 1B POSE", "NLLB 1B"],
            "title": "mT5 vs NLLB 1B",
        },
        {
            "experiment_names": ["TranslationMT53BExperiment", "TranslationIncompleteMT53BExperiment", "TranslationNLLB3BExperiment"],
            "legend_labels": ["mT5 3B baseline", "mT5 3B POSE", "NLLB 3B"],
            "title": "mT5 vs NLLB 3B",
        },
    ],
    title="Comparing mT5 baseline, mT5 POSE, and NLLB on Classical Tibetan to English",
    property_name="eval/bleu_score",
    y_label="Validation set BLEU score",
    save_name="tib2eng",
    y_lim=[0, 38],
)
