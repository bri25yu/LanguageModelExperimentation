### Methods
We train for 25 epochs with an early stopping patience of 2.


# Results
NLLB test general test set results are from https://github.com/facebookresearch/fairseq/tree/nllb#multilingual-translation-models. Specifically, we take the `bod_Tibt-eng_Latn` chrf++ metrics.


Both the train and val loss values are approximate as the loss doesn't necessarily converge (especially for training loss). "LR" stands for learning rate.


## Baselines

|Model     |Training  |Dataset|LR  |Train loss|Val loss|Test loss|Test chrf++ score|
|----------|----------|-------|----|----------|--------|---------|-----------------|
|NLLB 54.5B|?         |NLLB   |N/A |N/A       |N/A     |N/A      |             38.8|
|NLLB 600M |?         |NLLB   |N/A |N/A       |N/A     |N/A      |             32.7|
|NLLB 600M |zero-shot |Ours   |N/A |N/A       |N/A     |N/A      |             17.6|
|NLLB 600M |fine-tuned|Ours   |1e-5|      1.36|    1.46|     1.47|             42.5|
|NLLB 600M |fine-tuned|Ours   |2e-5|      1.11|    1.31|     1.33|             45.9|
|NLLB 600M |fine-tuned|Ours   |3e-5|      0.94|    1.24|     1.27|             47.8|


## Attention driven dropout

Shared experiment parameters

|Parameter|Value     |
|---------|----------|
|Model    |NLLB 600M |
|Training |fine-tuned|
|Dataset  |Ours      |

Results

We ablate over different values of attention dropout, denoted as p.

|Experimental config|LR  |Train loss|Val loss|Test loss|Test chrf++ score|
|-------------------|----|----------|--------|---------|-----------------|
|Baseline best      |3e-5|      0.94|    1.24|     1.27|             47.8|
|Attn Dropout p=0.05|1e-5|      1.34|    1.44|     1.46|             42.7|
|Attn Dropout p=0.05|2e-5|      1.08|    1.30|     1.32|             46.0|
|Attn Dropout p=0.05|3e-5|      0.91|    1.25|     1.27|             47.9|
|Attn Dropout p=0.10|1e-5|      1.35|    1.45|     1.47|             42.7|
|Attn Dropout p=0.10|2e-5|      |    |     |             |
|Attn Dropout p=0.10|3e-5|      |    |     |             |
|Attn Dropout p=0.15|1e-5|      |    |     |             |
|Attn Dropout p=0.15|2e-5|      |    |     |             |
|Attn Dropout p=0.15|3e-5|      |    |     |             |
