# Background
Original attention algorithm
1. Some definitions:
    1. N = batch size, L = sequence length, D = hidden dim, N_h = number of attention heads (for multi-headed attention)
    2. D_h = hidden dim for attention heads = D / N_h
    3. Input X of shape (N, L, D)
    4. Attention weights Q, K, V of shapes (D, D)
2. Q_L = XQ of shape (N, L, D). Reshape Q_L to (N, L, N_h, D_h) then to (N, N_h, L, D_h). Repeat the same process with K_L and V_L
3. attention_scores = Q_L V_L^T of shape (N, N_h, L, L)
4. attention_probs = softmax(attention_scores, axis=-1) of shape (N, N_h, L, L)
5. attention_probs = dropout(attention_probs) with probability p
6. context = attention probs V_L of shape (N, N_h, L, D_h). Reshape context to (N, L, N_h, D_h) then to (N, L, D)


# Methods
We train for 25 epochs with an early stopping patience of 2.

## Attention driven dropout
In the original attention algorithm, we modify step 5
```
# Original attention algorithm
5. attention_probs = dropout(attention_probs) with probability p
```
to
```
# Dropout argmax score in addition to regular dropout
5. attention_probs = dropout(attention_probs) with probability p
    1. attention_probs = dropout(attention_probs.argmax()) with probability p_argmax
```

We ablate over different argmax attention prob dropout values i.e. `p_argmax` of 5%, 10%, 15%, 25%, and 50%. There was no difference between the performance of the baseline with the performance of this method.

Why does dropping out the argmax attention value during training even with a high probability of 50% change so little? Part of it has to do with the scale of the attention probabilities, specifically that the dropout matrix is shaped (L, L) where L=100 is the sequence length and dropping out one value doesn’t actually change much. 

So, instead of dropping out a single value, we apply dropout with probability weighted by the attention score. Higher attention score values have a higher probability to be dropped out. We ablate over p = 5%, 10%, and 15%.

However, there was no difference between the performance of this technique and the baseline. The expected number of values dropped out is p * L * L where L is the sequence length. But if we weight by our attention probs, this value drops to dropout * L * 1. So, in theory, we need to multiply our dropout probability by a factor of L. We ablate over just 5%. We observe that the performance degrades significantly, where the train loss is much much higher than the baseline train loss, a sign that we're dropping out too much during training.

As a result, instead of multiplying by L to recover the true expected value, we multiply by sqrt(L) instead.


## LoRA
We apply the Low-Rank Adaptation (LoRA) of Large Language Models (see https://arxiv.org/abs/2106.09685). 

First, we freeze all the pretrained model parameters. For our attention algorithm, we modify step 2. Specifically we change step 2 from
```
# Original attention algorithm
2. Q_L = XQ of shape (N, L, D). Reshape Q_L to (N, L, N_h, D_h) then to (N, N_h, L, D_h). Repeat the same process with K_L and V_L
```
to
```
2. Q_L = X(Q + Q_A Q_B) ... (the rest is the same)
```


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

We fine-tune over learning rates in {1e-5, 2e-5, 3e-5}.

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
|Attn Dropout p=0.05|1e-5|      |    |     |             |
|Attn Dropout p=0.05|2e-5|      |    |     |             |
|Attn Dropout p=0.05|3e-5|      |    |     |             |
|Attn Dropout p=0.10|1e-5|      |    |     |             |
|Attn Dropout p=0.10|2e-5|      |    |     |             |
|Attn Dropout p=0.10|3e-5|      |    |     |             |
|Attn Dropout p=0.15|1e-5|      |    |     |             |
|Attn Dropout p=0.15|2e-5|      |    |     |             |
|Attn Dropout p=0.15|3e-5|      |    |     |             |


## LoRA

We fine-tune over learning rates in {2e-4, 3e-4, 4e-4}.

Shared experiment parameters

|Parameter|Value     |
|---------|----------|
|Model    |NLLB 600M |
|Training |fine-tuned|
|Dataset  |Ours      |

Results

We ablate over different values of rank.

|Experimental config|LR  |Train loss|Val loss|Test loss|Test chrf++ score|
|-------------------|----|----------|--------|---------|-----------------|
|Baseline best      |3e-5|      0.94|    1.24|     1.27|             47.8|
|rank=1             |2e-4|      |    |     |             |
|rank=1             |3e-4|      |    |     |             |
|rank=1             |4e-4|      |    |     |             |
|rank=2             |2e-4|      |    |     |             |
|rank=2             |3e-4|      |    |     |             |
|rank=2             |4e-4|      |    |     |             |
|rank=4             |2e-4|      |    |     |             |
|rank=4             |3e-4|      |    |     |             |
|rank=4             |4e-4|      |    |     |             |
|rank=8             |2e-4|      |    |     |             |
|rank=8             |3e-4|      |    |     |             |
|rank=8             |4e-4|      |    |     |             |
|rank=64            |2e-4|      |    |     |             |
|rank=64            |3e-4|      |    |     |             |
|rank=64            |4e-4|      |    |     |             |
