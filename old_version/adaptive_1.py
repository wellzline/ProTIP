import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
import logging
logging.basicConfig(
    filename='adaptive_log/log_test_1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from diffusers import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as transforms
from scipy import stats
import Levenshtein
import numpy as np
import time
import pandas as pd
torch.cuda.empty_cache()
from config import (
    e_threshold, ori_prompt, sigma, data_path,
    num_inference_steps, num_batch, batch_size,
    model_id
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
        

def cal_loss(ori_loss, disturb_prompt, ori_prompt):
    print("disturb_prompt", disturb_prompt)
    print("ori_prompt", ori_prompt)
    logging.info(f"dis_prompt: {disturb_prompt}")
    logging.info(f"ori_prompt: {ori_prompt}")
    logging.info(f"@" * 20)
    for i in range(num_batch):  # 5
        ori_interim_loss = ori_loss[batch_size*i : batch_size*(i+1)]
        generator = torch.Generator(device).manual_seed(1023+i)
        images = pipe([disturb_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator)

        dis_interim_loss = []
        for j in range(batch_size):  # 5
            dis_interim_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))

        _, p_value_ori = shapiro(ori_interim_loss)
        _, p_value_dis = shapiro(dis_interim_loss)
        logging.info(f"p_value_ori: {p_value_ori}")
        logging.info(f"p_value_dis: {p_value_dis}")
        if p_value_ori > 0.05 and p_value_dis > 0.05:  # normal distribution
            if stats.levene(dis_interim_loss, ori_loss).pvalue > 0.05:  # Consistent variance
                _, p_val = stats.ttest_ind(dis_interim_loss, ori_interim_loss, equal_var=True)
            else:
                _, p_val = stats.ttest_ind(dis_interim_loss, ori_interim_loss, equal_var=False)
            logging.info(f"normal distribution")
        else:
            _, p_val = mannwhitneyu(dis_interim_loss, ori_interim_loss)
            logging.info(f"non-normal distribution")
        
        logging.info(f"p_val: {p_val}")
        if p_val <= 0.0158:
            logging.info(f"interim num : {i+1}")
            return 0    
        logging.info(f"@" * 20)
    logging.info(f"interim num : {num_batch}")   
    return 1
    

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


if __name__ == "__main__":
    start_time = time.time()
    logging.info(f"num_inference_steps: {num_inference_steps}")
    logging.info(f"num_batch: {num_batch}")
    logging.info(f"batch_size: {batch_size}")
    df = pd.read_csv(data_path)

    ori_loss = []
    for i in range(num_batch):
        generator = torch.Generator(device).manual_seed(1023+i)
        images = pipe([ori_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
        for j in range(batch_size):
            ori_loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
    logging.info(f"ori_loss: {len(ori_loss)} {ori_loss}")
    logging.info(f"*" * 120)
    for id in range(2, 5):
        E_n, n = 0, 0
        L_distance, AdvSt2i = [], []
        robust_re, epsilon_re = [], []
        sample_data = list(df[f"Column {id}"].dropna())
        logging.info(f"disturb rate: {id}")
        logging.info(f"disturb_num: {sample_data[0]}")
        prompt_2 = get_AE(sample_data)
        for i, disturb_prompt in enumerate(prompt_2):
            n = i + 1
            L_distance.append(Levenshtein.distance(ori_prompt, disturb_prompt))
            if cal_loss(ori_loss, disturb_prompt, ori_prompt):
                E_n += 1
            logging.info(f"E_n: {E_n}")
            robust, epsilon = calculate_R(E_n, n)
            robust_re.append(robust)
            epsilon_re.append(epsilon)
            logging.info(f"robust reach: {robust}")
            logging.info(f"epsilon reach: {epsilon}")
            logging.info(f"n reach: {n}")
            if epsilon <= e_threshold:
                logging.info(f"*" * 120)
                logging.info(f"n reach: {n}")
                logging.info(f"robust reach: {robust}")
                break
            print("*" * 120)
            logging.info(f"*" * 120)
        print("*" * 120)
        logging.info(f"robust = {robust_re}")
        logging.info(f"epsilon = {epsilon_re}")
        logging.info(f"E_n = {E_n}")
        logging.info(f"AdvSt2i = {round(np.mean(AdvSt2i), 2)}")
        logging.info(f"OriSt2i = {round(np.mean(ori_loss), 2)}")
        logging.info(f"Levenshtein = {round(np.mean(L_distance), 2)}")
        logging.info(f"robust = {robust}")
        logging.info(f"epsilon = {epsilon}")
        logging.info(f"n = {n}")


        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        logging.info(f"time cost: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        logging.info(f"&" * 150)





