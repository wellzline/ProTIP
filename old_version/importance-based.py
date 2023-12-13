import logging
import time
logging.basicConfig(
    filename='./log/log_importance_insertion_1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
from PIL import Image
import numpy as np
from scipy import stats
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import mahalanobis
import torch
import clip
from PIL import Image

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
transformation = CompositeTransformation([WordSwapRandomCharacterInsertion(),WordSwapRandomCharacterDeletion(),WordSwapQWERTY(),WordSwapRandomCharacterSubstitution(),WordSwapNeighboringCharacterSwap()])
constraints = [RepeatModification(), StopwordModification()]

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

num_inference_steps = 50
num_batch = 5  
batch_size = 3

logging.info(f"num_inference_steps: {num_inference_steps}")
logging.info(f"num_batch: {num_batch}")
logging.info(f"batch_size: {batch_size}")

from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.ToTensor(),
    ])
metric = CLIPScore().to(device)
def calculate_text_image_distance(text, image):
    # print(transform(image).shape)  # torch.Size([3, 512, 512])
    img = transform(image)*255
    score = metric(img.to(device),text)
    return score.detach().cpu().numpy().item()
        
def cal_loss(flag, prompt, ori_prompt):
    print("dis_prompt:", prompt)
    print("ori_prompt:", ori_prompt)
    logging.info(f"dis_prompt: {prompt}")
    logging.info(f"ori_prompt: {ori_prompt}")
    loss = []
    cosine_group = []
    if flag:
        for i in range(num_batch):
            generator = torch.Generator(device).manual_seed(1023+i)
            images = pipe([prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
            for j in range(batch_size):
                loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
                images.images[j].save(f"f1/image_{i*3+j}.png")
        return loss, 0
    else:
        for i in range(num_batch):
            generator = torch.Generator(device).manual_seed(1023+i)
            images = pipe([prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
            for j in range(batch_size):
                loss.append(calculate_text_image_distance(ori_prompt,images.images[j]))
                
                image_1 = Image.open(f"f1/image_{i*3+j}.png")
                image_features_1 = transform(image_1)*255
                image_features_2 = transform(images.images[j])*255 
                cosine_similarity = np.dot(image_features_1.flatten(), image_features_2.flatten()) / (np.linalg.norm(image_features_1.flatten()) * np.linalg.norm(image_features_2.flatten()))
                cosine_group.append(cosine_similarity)
                images.images[j].save(f"f2/image_{str(cosine_similarity)}.png")
        print("cosine_group:", len(cosine_group), np.sum(np.array(cosine_group) > 0.9), cosine_group)
        logging.info(f"cosine_group {len(cosine_group)} {np.sum(np.array(cosine_group) > 0.9)}, {cosine_group}")
        return loss, np.sum(np.array(cosine_group) > 0.9)



#把[0:1]以k作为中点用指数放缩到新的[0:1], 例如k是0.95时输入0.95得到0.5
def scale(x, k):
    x = x**0.1
    if x<=k:
        return 0.5*x/k
    else:
        return 0.5+0.5*(x-k)/(1-k)

import string

def generate_insert_perturbations(sentence, target_words):
    words = sentence.split()
    perturbed_sentences = []
    for target_word in target_words:
        for insertion_point in range(len(target_word) + 1):
            for inserted_char in string.ascii_lowercase:
                perturbed_word = target_word[:insertion_point] + inserted_char + target_word[insertion_point:]
                perturbed_sentence = ' '.join(words).replace(target_word, perturbed_word, 1)
                perturbed_sentences.append(perturbed_sentence)
    return perturbed_sentences


if __name__ == "__main__":
    start_time = time.time()
    x, y, z = [], [], []
    ori_prompt_1 = "A red ball on green grass under a blue sky."
    # ori_prompt = "a photograph of an astronaut riding a white horse on the moon."
    # ori_prompt_2 = "A blue bench under a tree in the park surrounded by red leaves."
    ori_prompt = ori_prompt_1
    disturb_prompt = ori_prompt
    ori_loss, co = cal_loss(1, disturb_prompt, ori_prompt)
    # words = ori_prompt_1.split()
    # prompt_2 = [' '.join(words[:i] + words[i+1:]) for i in range(len(words))]
    # target_words = ["A", "on", "green", "a", "blue", "sky"]
    target_words = ["red", "ball", "green", "grass", "under"]
    prompt_2 = generate_insert_perturbations(ori_prompt_1, target_words)
    logging.info(f"disturb_num: {len(prompt_2)}")
    p_values = []
    count = 0
    result_dict = {
        'P>=0.6': [],
        'P>=0.7': [],
        'P>=0.8': [],
        'P>=0.9': [],
    }
    for i, disturb_prompt in enumerate(prompt_2):
        dis_loss, co = cal_loss(0, disturb_prompt, ori_prompt)
        print('dis_loss:', len(dis_loss), dis_loss)
        print('ori_loss:', len(ori_loss), ori_loss)
        logging.info(f"dis_loss: {len(dis_loss)} {dis_loss}")
        logging.info(f"ori_loss: {len(ori_loss)} {ori_loss}")
        # print("stats.levene(loss, ori_loss).pvalue:", stats.levene(loss, ori_loss).pvalue)
        if stats.levene(dis_loss, ori_loss).pvalue > 0.05:  # 方差一致
            t_stat, p_val = stats.ttest_ind(dis_loss, ori_loss, equal_var=True)
        else:
            t_stat, p_val = stats.ttest_ind(dis_loss, ori_loss, equal_var=False)
        print("p_val:", p_val)
        logging.info(f"p_val: {p_val}")
        prob = scale(p_val, 0.15)
        print("prob:", prob)
        logging.info(f"prob: {prob}")
        x.append(p_val)
        y.append(prob)
        z.append(co)
        if prob >= 0.6:
            result_dict['P>=0.6'].append(co)
        if prob >= 0.7:
            result_dict['P>=0.7'].append(co)
        if prob >= 0.8:
            result_dict['P>=0.8'].append(co)
        if prob >= 0.9:
            result_dict['P>=0.9'].append(co)
        count_dict = {'P>=0.6': {}, 'P>=0.7': {}, 'P>=0.8': {}, 'P>=0.9': {}}
        for condition, values in result_dict.items():
            for value in values:
                if value in count_dict[condition]:
                    count_dict[condition][value] += 1
                else:
                    count_dict[condition][value] = 1
        print("*" * 120)
        logging.info(f"*" * 120)

    print(count_dict)
    logging.info(f"count_dict: {count_dict}")
    sorted_count_dict = {
        'P>=0.6': dict(sorted(count_dict['P>=0.6'].items())),
        'P>=0.7': dict(sorted(count_dict['P>=0.7'].items())),
        'P>=0.8': dict(sorted(count_dict['P>=0.8'].items())),
        'P>=0.9': dict(sorted(count_dict['P>=0.9'].items())),
    }
    print(sorted_count_dict)
    print("P>=0.7 count:", len(result_dict['P>=0.7']))
    print("P>=0.8 count:", len(result_dict['P>=0.8']))
    print("x = ", x)
    print("y = ", y)
    logging.info(f"sorted_count_dict: {sorted_count_dict}")
    logging.info(f"P>=0.7 count: {len(result_dict['P>=0.7'])}")
    logging.info(f"P>=0.8 count: {len(result_dict['P>=0.8'])}")
    logging.info(f"P>=0.9 count: {len(result_dict['P>=0.9'])}")
    logging.info(f"x =  {x}")
    logging.info(f"y =  {y}")
    logging.info(f"z =  {z}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"程序运行时间：{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    logging.info(f"程序运行时间：{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
