
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
import random
seed_value = 42
random.seed(seed_value)
from textattack.transformations import (
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapRandomCharacterDeletion,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,  
)
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import Augmenter
# transformation = CompositeTransformation([WordSwapRandomCharacterInsertion(),WordSwapRandomCharacterSubstitution(),WordSwapRandomCharacterDeletion(),WordSwapNeighboringCharacterSwap(),WordSwapQWERTY()])
transformation = CompositeTransformation([WordSwapQWERTY()])
constraints = [RepeatModification()]

import pandas as pd
import clip
import numpy as np
from tqdm import tqdm
model, preprocess = clip.load("ViT-B/32", device=device)

def get_data(path):
    data = {}
    i = 1
    with open(path,'r') as file:
        for line in file:
            data[i] = line.strip()
            i += 1
    return data

if __name__ == "__main__":
    df = pd.DataFrame()
    prompt = "A red ball on green grass under blue sky"

    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1, transformations_per_example=10)

    prompt_2 = augmenter.augment(prompt)
    print("prompt_2:", prompt_2)