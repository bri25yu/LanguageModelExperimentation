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

conda create -y -n LanguageModelExperimentation python=3.9

conda activate LanguageModelExperimentation

# The CUDA Toolkit is split between two conda packages:
#   cudatoolkit - includes CUDA runtime support
#   cudatoolkit-dev - includes the CUDA compiler, headers, etc. needed for application development
conda install -y -c conda-forge cudatoolkit-dev=11.4.0
conda install -y pytorch cudatoolkit=11.3 -c pytorch

pip -q -q install -r attention_driven/requirements.txt
pip install -e .

deepspeed run.py

python attention_driven/read_results.py
```
