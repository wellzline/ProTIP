'''
The Gramformer project for GEC

https://github.com/PrithivirajDamodaran/Gramformer

pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
python -m spacy download en_core_web_sm

'''
from gramformer import Gramformer
import torch
from tqdm import tqdm

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

with open("/home/zhang2_y@WMGDS.WMG.WARWICK.AC.UK/workspace/robustness/discrete/generate_AE/char_AE/result_1_10.txt", 'r') as file:
    lines = file.readlines()
    strings = [line.split(':')[0].strip() for line in lines]

influent_sentences = strings
  
print("origin:", len(influent_sentences))

count = 0
for i, influent_sentence in tqdm(enumerate(influent_sentences[0:20])):
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    print("[Input] ", influent_sentence)
    for corrected_sentence in corrected_sentences:
        print("[Correction] ",corrected_sentence)
        if corrected_sentence == "A sunny beach with a palm tree and a surfboard lying in the sand.":
            count += 1
print("count:", count)
