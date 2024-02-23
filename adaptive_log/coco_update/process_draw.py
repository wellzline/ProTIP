import os
from scipy.stats import mannwhitneyu
from scipy import stats
import numpy as np
from scipy.stats import shapiro

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

def calculate_R(E_n, n):
    import math
    sigma = 0.05
    robust_left, robust_right, epsilon = 0, 0, 0
    epsilon = math.sqrt( (0.6 * math.log(math.log(n, 1.1) + 1, 10) + (1.8 ** -1) * math.log(24/sigma, 10)) / n )
    robust_left = E_n/n - epsilon
    robust_right = E_n/n + epsilon
    return robust_left, robust_right, epsilon

def process_full(file_path, file_result):
    Non_AE, AE = 0, 0
    with open(file_path, 'r') as file, open('./result.txt', 'w') as file_w:
        content = file.readlines()
        for i in range(len(content)):
            if 'ori_loss' in content[i]:
                line = content[i].split('[')[1]
                file_w.write(line[:-2] + '\n')

            if 'dis_interim_loss: 60; [' in content[i]:
                line = content[i].split('[')[1]
                file_w.write(line[:-2] + '\n')

    with open('result.txt', 'r') as file:
        content = file.readlines()

    ori_interim_loss = [float(item) for item in content[0].strip().split(',')]

    for n in range(1, len(content)):
        dis_interim_loss = [float(item) for item in content[n].strip().split(',')]
        for j in range(5):
            _, p_1 = shapiro(ori_interim_loss[0: 12*(j+1)])
            _, p_2 = shapiro(dis_interim_loss[0: 12*(j+1)]) 
            if p_1 > 0.05 and p_2 > 0.05:  # normal distr
                p_val, z_val = calculate_t_zValue(ori_interim_loss[0: 12*(j+1)], dis_interim_loss[0: 12*(j+1)])
            else:
                p_val, z_val = calculate_u_zValue(ori_interim_loss[0: 12*(j+1)], dis_interim_loss[0: 12*(j+1)])

            if z_val >= efficacy_boundary[j]:
                AE += 1
                break
            
            if z_val <= futility_boundary[j]:
                Non_AE += 1
                break
            
            if j == 4:
                if p_val > alpha[j]:
                    Non_AE += 1
                else:
                    AE += 1
            
        robust_left, robust_right, epsilon = calculate_R(Non_AE, n)
        file_result.write(str(robust_left) + ';')
    return 

if __name__ == "__main__":
    alpha = [0.0148, 0.0262, 0.0354, 0.0432, 0.05]
    futility_boundary = [-0.145, 0.511, 1.027, 1.497, float('inf')]
    efficacy_boundary = [2.176, 2.144, 2.113, 2.090, 2.071]  # critical value when Z-score > critical value, the reject the H0
    folders = ['./10_rate/70/', './10_rate/80/', './10_rate/90/']

    file_result = open(f"./result/defence/10_rate/SDV15_10_spellchecker.txt", "w")  
    file_paths = []
    for index in range(1, 37):
        file_paths.append(f"./10_rate/defence/spellchecker/log_char_{index}.log")
    print(file_paths)
    for i in range(0, 36):
        print(f'process: {file_paths[i]}')
        with open(file_paths[i], 'r') as file:
            for line in file:
                if 'robust reach:' in line:
                    robust_left = line.strip().split(':')[-1].split(',')[0].strip()
                    file_result.write(robust_left + ';')
        file_result.write('\n')
        # if i < 23:
        #     process_full(file_paths[i], file_result)
        #     file_result.write('\n')
        # else:
        #     with open(file_paths[i], 'r') as file:
        #         for line in file:
        #             if 'robust reach:' in line:
        #                 robust_left = line.strip().split(':')[-1].split(',')[0].strip()
        #                 file_result.write(robust_left + ';')
        #     file_result.write('\n')

    file_result.close()


'''
# process prompt5 defence autocorrect
with open(f"./10_rate/defence/autocorrect/log_char_5.log", 'r') as file, open('1.txt','w') as file_result:
    for line in file:
        if 'robust reach:' in line:
            robust_left = line.strip().split(':')[-1].split(',')[0].strip()
            file_result.write(robust_left + ';')
    file_result.write('\n')
'''





