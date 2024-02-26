'''
The Gramformer project for GEC

https://github.com/PrithivirajDamodaran/Gramformer

pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
python -m spacy download en_core_web_sm

'''
from gramformer import Gramformer
import torch
from tqdm import tqdm
import pandas as pd
gf = Gramformer(models = 1, use_gpu=False) 

df = pd.read_csv('./result_1.csv')
sample_data = list(df[f"Column {1}"].dropna())
strings = [line.split(':')[0].strip() for line in sample_data[1:]]
print(len(strings))

influent_sentences = strings
  
print("origin:", len(influent_sentences))

count = 0
for i, influent_sentence in enumerate(influent_sentences):
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    print("[Input] ", influent_sentence)
    for corrected_sentence in corrected_sentences:
        print("[Correction] ",corrected_sentence)
        if corrected_sentence == "A sunny beach with a palm tree and a surfboard lying in the sand.":
            count += 1
print("count:", count)
