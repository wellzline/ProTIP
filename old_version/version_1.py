from PIL import Image
import numpy as np
from scipy import stats
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import mahalanobis
import torch
import clip
from PIL import Image
from generate import generator_discrete

# import transformations, contraints, and the Augmenter
from textattack.transformations import (
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterSubstitution,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,   # 键盘错误 come from typing too quickly.  
)
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import Augmenter
# Set up transformation using CompositeTransformation()
transformation = CompositeTransformation([WordSwapRandomCharacterInsertion()])
# Set up constraints
constraints = [RepeatModification(), StopwordModification()]
count_90, count_95 = 0, 0

if __name__ == "__main__":
    num_images = 50
    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1, transformations_per_example=100)
    prompt_1 = 'A red ball on green grass.'
    prompt_2 = augmenter.augment(prompt_1)
    p_values = []
    generator_discrete(1, prompt_1 = prompt_1, prompt_2 = "", num_images = num_images)

    for i, item in enumerate(prompt_2):
        generator_discrete(0, prompt_1 = "", prompt_2 = item, num_images = num_images) 
        images_group1, images_group2 = [], []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        with torch.no_grad():
            text_features_1 = model.encode_text(clip.tokenize([prompt_1]).to(device))
            text_features_2 = model.encode_text(clip.tokenize([item]).to(device))
        
        cosine_group = []
        group_1, group_2 = [], []
        for i in range(num_images):
            image_1 = preprocess(Image.open(f"f1/image_{i}.png")).unsqueeze(0).to(device)
            image_2 = preprocess(Image.open(f"f2/image_{i}.png")).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features_1 = model.encode_image(image_1)
                image_features_2 = model.encode_image(image_2)

                group_1.append(torch.mm(image_features_1, text_features_1.t()).cpu().numpy().item())
                group_2.append(torch.mm(image_features_2, text_features_1.t()).cpu().numpy().item())

            image_features_1 = image_features_1.cpu().numpy()
            image_features_2 = image_features_2.cpu().numpy()

            cosine_similarity = np.dot(image_features_1.flatten(), image_features_2.flatten()) / (np.linalg.norm(image_features_1.flatten()) * np.linalg.norm(image_features_2.flatten()))
            cosine_group.append(cosine_similarity)

        t_stat, p_value = stats.ttest_ind(group_1, group_2)
        p_values.append(p_value)
        print("prompt_1:", prompt_1)
        print("prompt_2:", item)
        count_90 +=1 if np.sum(np.array(cosine_group) > 0.9) >=5 else 0
        count_95 +=1 if np.sum(np.array(cosine_group) > 0.95) >=5 else 0
        print("cosine similarity:", np.sum(np.array(cosine_group) > 0.9)/50, np.sum(np.array(cosine_group) > 0.9), np.sum(np.array(cosine_group) > 0.95), cosine_group)
        print("t-test:", p_value)
        print("*" * 120)

print("count_90:", count_90, count_90/len(prompt_2))
print("count_95:", count_95, count_95/len(prompt_2))
print("p_values:", np.sum(np.array(p_values) >= 0.05), np.sum(np.array(p_values) >= 0.05)/len(prompt_2), p_values)






