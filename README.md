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

conda env create -f environment.yml

conda activate LME

pip install -e .

deepspeed run.py

python read_results.py
```
