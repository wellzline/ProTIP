

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
import logging

from scipy.stats import mannwhitneyu
from diffusers import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as transforms
import Levenshtein
import numpy as np
import time
import pandas as pd
torch.cuda.empty_cache()
from config import (
    e_threshold, origin_prompt_path, sigma, data_path,
    num_inference_steps, num_batch, batch_size,
    model_id, ori_prompt, stop_early
)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

transform = transforms.Compose([transforms.ToTensor()])
metric = CLIPScore().to(device)
def calculate_text_image_distance(text, image):
    # print(transform(image).shape)  # torch.Size([1, 512, 512])
    img = transform(image)*255
    score = metric(img.to(device), text)
    return score.detach().cpu().numpy().item()

def setup_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger  

def cal_loss(ori_loss, disturb_prompt, ori_prompt):
    global stop_early
    stop_early = 0
    print("dis_prompt", disturb_prompt)
    print("ori_prompt", ori_prompt)
    logger.info(f"dis_prompt: {disturb_prompt}")
    logger.info(f"ori_prompt: {ori_prompt}")
    logger.info(f"@" * 20)
    dis_interim_loss = []
    for i in range(num_batch):  # 5
        ori_interim_loss = ori_loss[0: batch_size*(i+1)]
        generator = torch.Generator(device).manual_seed(1023+i)
        images = pipe([disturb_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator)
        for j in range(batch_size):  # 5
            dis_interim_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
        
        _, p_val = mannwhitneyu(dis_interim_loss, ori_interim_loss)
        logger.info(f"p_val: {p_val}")
        if p_val <= 0.0158:
            if i <= 3:
                stop_early = 1
            logger.info(f"Reject interim num : {i+1}")
            return 0, dis_interim_loss
        logger.info(f"@" * 20)
    logger.info(f"Accept interim num : {num_batch}")   
    return 1, dis_interim_loss
    

def get_AE(sample_data):
    import random
    random.seed(42) 
    strings = [line.split(':')[0].strip() for line in sample_data[1:]]
    sampled_strings = random.sample(strings, len(strings))
    return sampled_strings

def calculate_R(E_n, n):
    import math
    robust, epsilon = 0, 0
    epsilon = math.sqrt( (0.6 * math.log(math.log(n, 1.1) + 1, 10) + (1.8 ** -1) * math.log(24/sigma, 10)) / n )
    robust = (E_n - epsilon) / n
    return robust, epsilon

def get_origin_prompt(origin_prompt_path):
    origin_prompt = {}
    i = 1
    with open(origin_prompt_path,'r') as file:
        for line in file:
            origin_prompt[i] = line.strip()
            i += 1
    return origin_prompt

if __name__ == "__main__":
    start_time = time.time()
    
    origin_prompts = get_origin_prompt(origin_prompt_path) 
    
    for index, ori_prompt in origin_prompts.items():
        logger = setup_logger(f"adaptive_log/log_{index}.log")
        logger.info(f"num_inference_steps: {num_inference_steps}")
        logger.info(f"num_batch: {num_batch}")
        logger.info(f"batch_size: {batch_size}")
        df = pd.read_csv(f"./generate_AE/char_AE/result_{index}.csv")
        logger.info(f"./generate_AE/char_AE/result_{index}.csv")
        efficient_n = 0
        if index <=3:
            pass
        else:
            logger.info(f"ori_prompt: {ori_prompt}")
            ori_loss = []
            for i in range(num_batch):
                generator = torch.Generator(device).manual_seed(1023+i)
                images = pipe([ori_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
                for j in range(batch_size):
                    ori_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
            logger.info(f"ori_loss: {len(ori_loss)} {ori_loss}")
            logger.info(f"*" * 120)
            for id in range(1, 5):
                efficient_n = 0
                E_n, n = 0, 0
                L_distance, AdvSt2i = [], []
                robust_re, epsilon_re = [], []
                prompt_2 = [f"{index}A mysterious and magical forest with glowing mushrooms and hidden creatures.", "Serene meadow with wildflowers swaying in the gentle breeze under a clear blue sky."]
                logger.info(f"disturb rate: {id}")
                # logger.info(f"disturb_num: {sample_data[0]}")
                # prompt_2 = get_AE(sample_data)
                for i, disturb_prompt in enumerate(prompt_2):
                    n = i + 1
                    L_distance.append(Levenshtein.distance(ori_prompt, disturb_prompt))
                    whether_robust, dis_loss = cal_loss(ori_loss, disturb_prompt, ori_prompt)
                    if whether_robust:
                        E_n += 1
                    if stop_early:
                        efficient_n += 1
                    AdvSt2i.append(sum(dis_loss) / len(dis_loss))
                    robust, epsilon = calculate_R(E_n, n)
                    robust_re.append(robust)
                    epsilon_re.append(epsilon)
                    logger.info(f"stop_early: {efficient_n}")
                    logger.info(f"E_n: {E_n}")
                    logger.info(f"n: {n}")
                    logger.info(f"robust reach: {robust}")
                    logger.info(f"epsilon reach: {epsilon}")
                    if epsilon <= e_threshold:  # stop condition
                        break
                    print("*" * 120)
                    logger.info(f"*" * 120)
                print("*" * 120)
                logger.info(f"*" * 120)
                logger.info(f"robust = {robust_re}")
                logger.info(f"epsilon = {epsilon_re}")
                logger.info(f"stop_early: {efficient_n}")
                logger.info(f"E_n = {E_n}")
                logger.info(f"n = {n}")
                logger.info(f"AdvSt2i = {round(np.mean(AdvSt2i), 2)}")
                logger.info(f"OriSt2i = {round(np.mean(ori_loss), 2)}")
                logger.info(f"Levenshtein = {round(np.mean(L_distance), 2)}")
                logger.info(f"robust = {robust}")
                logger.info(f"epsilon = {epsilon}")
                


                end_time = time.time()
                elapsed_time = end_time - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
                logger.info(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
                logger.info(f"&" * 150)





