# LanguageModelExperimentation

### Example setup
```bash
# Install conda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

conda env create -f environment.yml
conda activate LME

deepspeed run.py

python scripts/read_results.py
```


### Max batch sizes
|GPU size|Model      |Batch size|
|--------|-----------|----------|
|    24GB|  NLLB 600M|        16|
|        |    NLLB 1B|         4|
|        |   mT5 600M|         8|
|        |     mT5 1B|         ?|
|        |  byT5 600M|         ?|
|        |    byT5 1B|         ?|
|        |FlanT5 300M|         ?|
|        |FlanT5 800M|         ?|
|        |  FlanT5 3B|         ?|
|    49GB|    NLLB 1B|        16|
|        |    NLLB 3B|        16|
|        |     mT5 1B|        16|
|        |     mT5 3B|         4|
|        |    mT5 13B|         ?|


### Ideas
- Activation function diversification
- Single layer model with many attention heads
