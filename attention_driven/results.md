### Results
NLLB test general test set results are from https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models. Specifically, we take the `bod_Tibt-eng_Latn` chrf++ metrics.


Both the train and val loss values are approximate as the loss doesn't necessarily converge (especially for training loss). "LR" stands for learning rate


Baselines: Mask prob = 0.0, Attention dropout = 0.1

|Model|Training|Dataset|LR|Train loss|Val loss|Test loss|Test chrf++ score|
|----------|----------|----|----|---|---|----|-----|
|NLLB 54.5B|?         |NLLB|N/A |N/A|N/A|N/A |38.8 |
|NLLB 600M |?         |NLLB|N/A |N/A|N/A|N/A |32.7 |
|NLLB 600M |zero-shot |Ours|N/A |N/A|N/A|N/A |17.56|
|NLLB 600M |fine-tuned|Ours|1e-5|1.7|1.5|1.68|38.06|
|NLLB 600M |fine-tuned|Ours|2e-5|1.3|1.4|1.50|41.63|
|NLLB 600M |fine-tuned|Ours|3e-5|1.3|1.3|1.41|43.55|
