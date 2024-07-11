
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# transformation = CompositeTransformation([WordSwapRandomCharacterSubstitution()])
constraints = [RepeatModification(), StopwordModification()]

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
    prompt = get_data('./origin_prompts/coco.txt')
    for id, input in prompt.items():
        if id == 5:
            result_data = []
            for rate in range(1, 3):
                augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=float("0." + str(rate)), transformations_per_example=1000000)
                print("input:", input)
                prompt_2 = augmenter.augment(input)
                print("len(prompt_2):", len(prompt_2))
                text = clip.tokenize([input]).to(device)
                result = {}
                with torch.no_grad():
                    text_features = model.encode_text(text)
                    for item in tqdm(prompt_2):
                        disturb = clip.tokenize([item]).to(device)
                        disturb_features = model.encode_text(disturb)
                        text_features = text_features.to('cpu', dtype=torch.float32)
                        disturb_features = disturb_features.to('cpu', dtype=torch.float32)
                        cosine_similarity = np.dot(text_features.flatten().numpy(), disturb_features.flatten().numpy()) / (np.linalg.norm(text_features.flatten().numpy()) * np.linalg.norm(disturb_features.flatten().numpy()))
                        result[item] = cosine_similarity

                sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
                filtered_items = {key: value for key, value in sorted_result.items() if value > 0.85}
                print("len(filtered_items):", len(filtered_items))
                data = [len(filtered_items)] + [f"{key}: {value}" for key, value in filtered_items.items()]
                df = pd.DataFrame({f'Column {rate}': data})
                result_data.append(df)

            result_df = pd.concat(result_data, axis=1)
            out_file = f"./coco/char_AE/result_{id}.csv"
            result_df.to_csv(out_file, index=False)
        
  




 
