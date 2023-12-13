import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
import logging
logging.basicConfig(
    filename='adaptive_log/log_word_1_10_word_AE.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
from scipy.spatial.distance import mahalanobis
from diffusers import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
from scipy import stats
from PIL import Image
import Levenshtein
import numpy as np
import time
import pandas as pd
import clip
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
        
def cal_loss(disturb_prompt, ori_prompt):
    print("disturb_prompt", disturb_prompt)
    print("ori_prompt", ori_prompt)
    logging.info(f"dis_prompt: {disturb_prompt}")
    logging.info(f"ori_prompt: {ori_prompt}")
    loss = []
    for i in range(num_batch):
        generator = torch.Generator(device).manual_seed(1023+i)
        images = pipe([disturb_prompt] * batch_size, num_inference_steps = num_inference_steps, generator = generator) 
        for j in range(batch_size):
            loss.append(calculate_text_image_distance(ori_prompt, images.images[j]))
    return loss
    
#把[0:1]以k作为中点用指数放缩到新的[0:1], 例如k是0.95时输入0.95得到0.5
def scale(x, k):
    x = x**0.1
    if x<=k:
        return 0.5*x/k
    else:
        return 0.5+0.5*(x-k)/(1-k)

def get_AE(sample_data):
    import random
    random.seed(42) 
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
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

    disturb_prompt = ori_prompt
    ori_loss = cal_loss(disturb_prompt, ori_prompt)
    logging.info(f"*" * 120)
    # R_threshold = {1: 0.88, 2: 0.68, 3: 0.58, 4: 0.85}
    for id in range(1,5):
        count_1, count_2, n = 0, 0, 0
        L_distance, AdvSt2i = [], []
        robust_re, epsilon_re = [], []
        sample_data = list(df[f"Column {id}"].dropna())
        logging.info(f"disturb rate: {id}")
        logging.info(f"disturb_num: {sample_data[0]}")
        prompt_2 = get_AE(sample_data)
        for i, disturb_prompt in enumerate(prompt_2):
            L_distance.append(Levenshtein.distance(ori_prompt, disturb_prompt))
            dis_loss = cal_loss(disturb_prompt, ori_prompt)
            AdvSt2i.append(sum(dis_loss) / len(dis_loss))
            print('dis_loss:', len(dis_loss), dis_loss)
            print('ori_loss:', len(ori_loss), ori_loss)
            logging.info(f"dis_loss: {len(dis_loss)} {dis_loss}")
            logging.info(f"ori_loss: {len(ori_loss)} {ori_loss}")
            # print("stats.levene(loss, ori_loss).pvalue:", stats.levene(loss, ori_loss).pvalue)
            if stats.levene(dis_loss, ori_loss).pvalue > 0.05:  # 方差一致
                t_stat, p_val = stats.ttest_ind(dis_loss, ori_loss, equal_var=True)
            else:
                t_stat, p_val = stats.ttest_ind(dis_loss, ori_loss, equal_var=False)
            prob = scale(p_val, 0.15)
            logging.info(f"p_val: {p_val}")
            logging.info(f"prob: {prob}")
            count_1 +=1 if p_val>= 0.05 else 0
            count_2 +=1 if prob>= 0.8 else 0
            logging.info(f"E_n: {count_1}")
            logging.info(f"count_2: {count_2}")
            n = i + 1
            robust, epsilon = calculate_R(count_1, n)
            robust_re.append(robust)
            epsilon_re.append(epsilon)
            logging.info(f"robust reach: {robust}")
            logging.info(f"epsilon reach: {epsilon}")
            logging.info(f"n reach: {n}")
            # if robust >= R_threshold[id] and epsilon <= e_threshold:
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
        logging.info(f"E_n = {count_1}")
        logging.info(f"count_2 = {count_2}")
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
        print(f"程序运行时间：{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        logging.info(f"程序运行时间：{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
        logging.info(f"&" * 150)

