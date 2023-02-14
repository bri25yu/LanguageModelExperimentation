### Max batch sizes
|GPU size|Model     |Batch size|
|--------|----------|----------|
|    24GB| NLLB 600M|        16|
|        |   NLLB 1B|         4|
|        |  mT5 600M|         8|
|        |    mT5 1B|         ?|
|        |Bloom 600M|        16|
|        |  Bloom 1B|         ?|
|    49GB|   NLLB 1B|        16|
|        |   NLLB 3B|        16|
|        |    mT5 1B|        16|
|        |    mT5 3B|         4|
|        |   mT5 13B|         ?|
|        |  Bloom 1B|        32|
|        |  Bloom 3B|         ?|
|        |  Bloom 7B|         ?|


### Ideas
- Activation function diversification
- Single layer model with many attention heads
