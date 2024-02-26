import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import logging
from scipy.stats import mannwhitneyu
from diffusers import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as transforms
import Levenshtein
from scipy.stats import shapiro
import numpy as np
import time
from scipy import stats
import pandas as pd
torch.cuda.empty_cache()
from config import (
    sample_num, origin_prompt_path, sigma,
    num_inference_steps, num_batch, batch_size,
    model_id
)
import random

def setup_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger  

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

transform = transforms.Compose([transforms.ToTensor()])
metric = CLIPScore().to(device)
def calculate_text_image_distance(text, image):
    img = transform(image)*255
    score = metric(img.to(device), text)
    return score.detach().cpu().numpy().item()

def calculate_u_zValue(data1, data2):
    U_statistic, p_val = mannwhitneyu(data1, data2, alternative='greater')
    n1, n2 = len(data1), len(data2)
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_value = (U_statistic - mean_U) / std_U
    return p_val, z_value

def calculate_t_zValue(data1, data2):
    t_statistic, p_val = stats.ttest_ind(data1, data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    z_value = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    return p_val, z_value


def cal_loss(ori_loss, disturb_prompt, ori_prompt):
    print("dis_prompt", disturb_prompt)
    print("ori_prompt", ori_prompt)
    logger.info(f"dis_prompt: {disturb_prompt}")
    logger.info(f"ori_prompt: {ori_prompt}")
    logger.info(f"--" * 20)
    dis_interim_loss = []
    alpha = [0.0148, 0.0262, 0.0354, 0.0432, 0.05]
    futility_boundary = [-0.145, 0.511, 1.027, 1.497, float('inf')]
    efficacy_boundary = [2.176, 2.144, 2.113, 2.090, 2.071] 
    for i in range(num_batch):  
        ori_interim_loss = ori_loss[0: batch_size*(i+1)]
        generator = torch.Generator(device).manual_seed(1023+i)
        images = pipe([disturb_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator)
        for j in range(batch_size):  
            dis_interim_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))

        logger.info(f"dis_interim_loss: {len(dis_interim_loss)}; {dis_interim_loss}")
        logger.info(f"ori_interim_loss: {len(ori_interim_loss)}; {ori_interim_loss}")
        _, p_1 = shapiro(ori_interim_loss[0: 12*(i+1)])
        _, p_2 = shapiro(dis_interim_loss[0: 12*(i+1)]) 
        if p_1 > 0.05 and p_2 > 0.05: 
            p_val, z_val = calculate_t_zValue(ori_interim_loss[0: 12*(i+1)], dis_interim_loss[0: 12*(i+1)])
        else:
            p_val, z_val = calculate_u_zValue(ori_interim_loss[0: 12*(i+1)], dis_interim_loss[0: 12*(i+1)])
        logger.info(f"p_val, z_val: {p_val} {z_val}")
        
        if z_val >= efficacy_boundary[i]:
            return 0
        
        if z_val <= futility_boundary[i]:
            return 1
        
        if i == 4:
            if p_val > alpha[i]:
                return 1
            else:
                return 0
        logger.info(f"--" * 20) 
    return 1
    

def get_AE(sample_data):
    import random
    random.seed(42) 
    strings = [line.split(':')[0].strip() for line in sample_data[1:]]
    sampled_strings = random.choices(strings, k=1)
    return sampled_strings

def calculate_R(E_n, n):
    import math
    robust_left, robust_right, epsilon = 0, 0, 0
    epsilon = math.sqrt( (0.6 * math.log(math.log(n, 1.1) + 1, 10) + (1.8 ** -1) * math.log(24/sigma, 10)) / n )
    robust_left = E_n/n - epsilon
    robust_right = E_n/n + epsilon
    return robust_left, robust_right, epsilon

def get_origin_prompt(origin_prompt_path):
    origin_prompt = {}
    i = 1
    with open(origin_prompt_path,'r') as file:
        for line in file:
            origin_prompt[i] = line.strip()
            i += 1
    return origin_prompt

def defence_gramfomer(influent_sentence):
    from gramformer import Gramformer
    gf = Gramformer(models = 1, use_gpu=False) 
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    for corrected_sentence in corrected_sentences:
        return corrected_sentence
    return ""

def defence_spellchecker(influent_sentence):
    from spellchecker import SpellChecker
    import pandas as pd
    spell = SpellChecker()
    str1 = influent_sentence
    str2 = str1.split()
    misspelled = spell.unknown(str1.split())  
    corrected_sentence = " ".join(spell.correction(word) if word in misspelled else word for word in str2 if spell.correction(word) is not None)
    return corrected_sentence + '.'

def defence_autocorrect(influent_sentence):
    from autocorrect import Speller
    spell = Speller(lang='en')
    text = influent_sentence
    corrected_text = ' '.join([spell(word) for word in text.split()])
    return corrected_text


if __name__ == "__main__":
    start_time = time.time()
    origin_prompts = get_origin_prompt(origin_prompt_path) 
    for index, ori_prompt in origin_prompts.items():
        if index == 19:
            AEdata_path = f"./generate_AE/coco/char_AE/result_{index}.csv"
            logger = setup_logger(f"adaptive_log/coco_update/spellchecker/log_char_{index}.log")
            logger.info(f"sigma: {sigma}")
            logger.info(f"num_inference_steps: {num_inference_steps}")
            logger.info(f"num_batch: {num_batch}")
            logger.info(f"batch_size: {batch_size}")
            logger.info(AEdata_path)
            logger.info(f"ori_prompt: {ori_prompt}")
            df = pd.read_csv(AEdata_path)
            ori_loss = []
            for i in range(num_batch):
                generator = torch.Generator(device).manual_seed(1023+i)
                images = pipe([ori_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
                for j in range(batch_size):
                    ori_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
            logger.info(f"ori_loss: {len(ori_loss)} {ori_loss}")
            logger.info(f"*" * 120)
            for id in range(1, 2): 
                efficient_n = 0
                Non_AE, n = 0, 0
                L_distance, AdvSt2i = [], []
                robust_re, epsilon_re = [], []
                sample_data = list(df[f"Column {id}"].dropna())
                strings = [line.split(':')[0].strip() for line in sample_data[1:]]
                logger.info(f"disturb rate: {id}")
                logger.info(f"disturb_num: {sample_data[0]}")
                n = 1
                epsilon = 1000
                for count in range(sample_num): 
                    selected = random.choices(strings, k=1)[0]
                    disturb_prompt = defence_spellchecker(selected)
                    if disturb_prompt == ori_prompt:
                        Non_AE += 1
                        logger.info(f"dis_prompt: {selected}")
                        logger.info(f"ori_prompt: {ori_prompt}")
                        logger.info(f"same")
                    else:
                        logger.info(f"unsame")
                        logger.info(f"selected: {selected}")
                        logger.info(f"revised: {disturb_prompt}")
                        L_distance.append(Levenshtein.distance(ori_prompt, disturb_prompt))
                        whether_robust = cal_loss(ori_loss, disturb_prompt, ori_prompt)
                        Non_AE += 1 if whether_robust else 0

                    robust_left, robust_right, epsilon = calculate_R(Non_AE, n)
                    robust_re.append((robust_left, robust_right))
                    epsilon_re.append(epsilon)
                    logger.info(f"stop_early: {efficient_n}")
                    logger.info(f"Non_AE: {Non_AE}")
                    logger.info(f"n: {n}")
                    logger.info(f"robust reach: {robust_left} , {robust_right}")
                    logger.info(f"epsilon reach: {epsilon}")
                    print("*" * 120)
                    logger.info(f"*" * 120)
                    n += 1
                print("*" * 120)
                logger.info(f"*" * 120)
                logger.info(f"robust = {robust_re}")
                logger.info(f"epsilon = {epsilon_re}")
                logger.info(f"stop_early = {efficient_n}")
                logger.info(f"E_n = {Non_AE}")
                logger.info(f"n = {n}")
                logger.info(f"AdvSt2i = {round(np.mean(AdvSt2i), 2)}")
                logger.info(f"OriSt2i = {round(np.mean(ori_loss), 2)}")
                logger.info(f"Levenshtein = {round(np.mean(L_distance), 2)}")
                logger.info(f"robust = {robust_left} , {robust_right}")
                logger.info(f"epsilon = {epsilon}")
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
                logger.info(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
                logger.info(f"&" * 150)


