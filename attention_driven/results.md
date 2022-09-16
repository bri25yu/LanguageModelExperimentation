### Methods
We train for 25 epochs with an early stopping patience of 2.


### Results
NLLB test general test set results are from https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models. Specifically, we take the `bod_Tibt-eng_Latn` chrf++ metrics.


Both the train and val loss values are approximate as the loss doesn't necessarily converge (especially for training loss). "LR" stands for learning rate.


Baselines

|Model     |Training  |Dataset|LR  |Train loss|Val loss|Test loss|Test chrf++ score|
|----------|----------|-------|----|----------|--------|---------|-----------------|
|NLLB 54.5B|?         |NLLB   |N/A |N/A       |N/A     |N/A      |             38.8|
|NLLB 600M |?         |NLLB   |N/A |N/A       |N/A     |N/A      |             32.7|
|NLLB 600M |zero-shot |Ours   |N/A |N/A       |N/A     |N/A      |             17.6|
|NLLB 600M |fine-tuned|Ours   |1e-5|          |        |         |                 |
|NLLB 600M |fine-tuned|Ours   |2e-5|          |        |         |                 |
|NLLB 600M |fine-tuned|Ours   |3e-5|          |        |         |                 |
