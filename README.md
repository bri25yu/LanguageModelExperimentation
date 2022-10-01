# LanguageModelExperimentation

### Example setup
```bash
# Clone data repository
git clone https://github.com/Linguae-Dharmae/language-models

# Clone this repository
git clone https://github.com/bri25yu/LanguageModelExperimentation

# Install conda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

cd LanguageModelExperimentation

# The CUDA Toolkit is split between two conda packages:
#   cudatoolkit - includes CUDA runtime support
#   cudatoolkit-dev - includes the CUDA compiler, headers, etc. needed for application development

# If you want to run t5 models on nvidia GPUs older running archs older than 2017 ampere arch, run this line
# This is transformers v4.22.0 with a special optional autocast for t5 models under fp16 precision
conda env create -f attention_driven/environment.yml

conda activate LME

pip install -e .

deepspeed run.py

python attention_driven/read_results.py
```
