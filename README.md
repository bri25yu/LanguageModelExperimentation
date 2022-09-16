# LanguageModelExperimentation

### Example setup
```bash
git clone https://github.com/Linguae-Dharmae/language-models

git clone https://github.com/bri25yu/LanguageModelExperimentation

cd LanguageModelExperimentation

conda create -n LanguageModelExperimentation --python=3.9

conda activate LanguageModelExperimentation

conda install pytorch cudatoolkit=11.3 -c pytorch
pip -q -q install -r attention_driven/requirements.txt
pip install -e .

deepspeed run.py

python attention_driven/read_results.py
```
