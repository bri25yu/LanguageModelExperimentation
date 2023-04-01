# LanguageModelExperimentation

Research conducted under Prof. Kurt Keutzer at Berkeley Artificial Intelligence Research (BAIR). 

<img src="http://bair.berkeley.edu/images/BAIR_Logo_BlueType_Tag.png" width="525" height="280">



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
For tib to eng translation:

|GPU size|Model      |Batch size|
|--------|-----------|----------|
|    24GB|  NLLB 600M|        16|
|        |    NLLB 1B|         4|
|        |   mT5 600M|         8|
|        |     mT5 1B|         4|
|    49GB|    NLLB 1B|        16|
|        |    NLLB 3B|        16|
|        |     mT5 1B|        16|
|        |     mT5 3B|         4|
|        |    mT5 13B|         ?|

For Flores200:

|GPU size|Precision|Model      |Seq len|Batch size|
|--------|---------|-----------|-------|----------|
|    24GB|     BF16|   mT5 300M|    128|        32|
|    24GB|     BF16|   mT5 300M|    256|        16|
|    24GB|     FP32|   mT5 300M|    128|         8|
|    24GB|     BF16|   mT5 600M|    256|         8|
|    24GB|     BF16|     mT5 1B|    256|         4|
|    48GB|     BF16|   mT5 300M|    128|        64|
|    48GB|     BF16|   mT5 300M|    128|        64|
|    48GB|     BF16|     mT5 1B|    256|        32|


### Ideas
- Activation function diversification
- Single layer model with many attention heads
