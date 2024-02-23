from scipy.stats import mannwhitneyu
from scipy import stats
import numpy as np
import logging
from scipy.stats import shapiro

def setup_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger 

def calculate_u_zValue(data1, data2):
    U_statistic, p_val = mannwhitneyu(data1, data2, alternative='greater')
    n1, n2 = len(data1), len(data2)
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_value = (U_statistic - mean_U) / std_U
    # with open('pz_value.txt', 'a+') as file:
    #     file.write(f"p_value: {p_val}, z_value: {z_value}\n")
    return p_val, z_value

def calculate_ks_zValue(data1, data2):
    ks_statistic, p_val = stats.ks_2samp(data1, data2)
    z_value = np.sqrt(len(data1) * len(data2) / (len(data1) + len(data2))) * ks_statistic
    return p_val, z_value

def calculate_t_zValue(data1, data2):
    t_statistic, p_val = stats.ttest_ind(data1, data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    z_value = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    # with open('pz_value.txt', 'a+') as file:
    #     file.write(f"p_value: {p_val}, z_value: {z_value}\n")
    return p_val, z_value

def calculate_R(E_n, n):
    import math
    sigma = 0.1
    robust_left, robust_right, epsilon = 0, 0, 0
    epsilon = math.sqrt( (0.6 * math.log(math.log(n, 1.1) + 1, 10) + (1.8 ** -1) * math.log(24/sigma, 10)) / n )
    robust_left = E_n/n - epsilon
    robust_right = E_n/n + epsilon
    return robust_left, robust_right, epsilon

if __name__ == "__main__":
    logger = setup_logger(f"./result/all_result.log")
    e_all, f_all = [], []
    e_all = {'1':[], '2':[], '3':[], '4':[], '5':[]}
    f_all = {'1':[], '2':[], '3':[], '4':[], '5':[]}
    efficacy_all = {'1':0, '2':0, '3':0, '4':0, '5':0}
    futility_all = {'1':0, '2':0, '3':0, '4':0, '5':0}
    for index in range(5, 6):
        with open(f"./10_rate/defence/autocorrect/log_char_{index}.log", 'r') as file, open('./result.txt', 'w') as file_w:
            content = file.readlines()
            for i in range(len(content)):
                if 'ori_loss' in content[i]:
                    line = content[i].split('[')[1]
                    file_w.write(line[:-2] + '\n')

                if "dis_interim_loss: 60;" in content[i]:
                    line = content[i].split('[')[1]
                    file_w.write(line[:-2] + '\n')

                if "len(dis_interim_loss): 60 60" in content[i]:
                    line = content[i-2].split('[')[1]
                    file_w.write(line[:-2] + '\n')
        print('Done!')

        with open('result.txt', 'r') as file:
            content = file.readlines()
        print(len(content))
        logger.info(f"prompt: {index}")
        logger.info(f"sample size: {len(content)}")

        alpha = [0.0148, 0.0262, 0.0354, 0.0432, 0.05]
        futility_boundary = [-0.145, 0.511, 1.027, 1.497, float('inf')]
        efficacy_boundary = [2.176, 2.144, 2.113, 2.090, 2.071]  # critical value when Z-score > critical value, the reject the H0
        Non_AE, AE = 0, 0
        efficacy_stages = {'1':0, '2':0, '3':0, '4':0, '5':0}
        futility_stages = {'1':0, '2':0, '3':0, '4':0, '5':0}
        x, y = 0, 0
        result = []
        ori_interim_loss = [float(item) for item in content[0].strip().split(',')]
        
        for n in range(1, len(content)):
            dis_interim_loss = [float(item) for item in content[n].strip().split(',')]
            for j in range(5):
                # p_val, z_val = calculate_ks_zValue(ori_interim_loss[0: 12*(j+1)], dis_interim_loss[0: 12*(j+1)])
                _, p_1 = shapiro(ori_interim_loss[0: 12*(j+1)])
                _, p_2 = shapiro(dis_interim_loss[0: 12*(j+1)]) 
                if p_1 > 0.05 and p_2 > 0.05:  # normal distr
                    p_val, z_val = calculate_t_zValue(ori_interim_loss[0: 12*(j+1)], dis_interim_loss[0: 12*(j+1)])
                else:
                    p_val, z_val = calculate_u_zValue(ori_interim_loss[0: 12*(j+1)], dis_interim_loss[0: 12*(j+1)])

                # if p_val <= alpha[j]:
                if z_val >= efficacy_boundary[j]:
                    x += 1
                    AE += 1
                    efficacy_stages[str(j+1)] += 1
                    y += 1 if z_val >= efficacy_boundary[j] else 0
                    # y += 1 if p_val <= alpha[j] else 0
                    break
                
                if z_val <= futility_boundary[j]:
                    Non_AE += 1
                    futility_stages[str(j+1)] += 1
                    break
                
                if j == 4:
                    if p_val > alpha[j]:
                        Non_AE += 1
                        futility_stages[str(j+1)] += 1
                    else:
                        AE += 1
                        efficacy_stages[str(j+1)] += 1
                
            robust_left, robust_right, epsilon = calculate_R(Non_AE, n)
            with open(f"result/{index}.txt", 'a+') as file:
                file.write(f"{robust_left};")

        with open('result/efficitive.txt', 'a+') as file:
            file.write(f"\n")
        print('AE:', AE)
        print('Non_AE:', Non_AE)
        print('efficacy:', sum(efficacy_stages.values()))
        print('futility:', sum(futility_stages.values()))
        print('efficacy_stages:', efficacy_stages)
        print('futility_stages:', futility_stages)
        print("evaluate PR:", Non_AE/400)
        print('x, y:', x, y)
        logger.info(f"AE: {AE}")
        logger.info(f"Non_AE: {Non_AE}")
        logger.info(f"efficacy: {sum(efficacy_stages.values())}")
        logger.info(f"futility: {sum(futility_stages.values())}")
        logger.info(f"efficacy_stages: {efficacy_stages}")
        logger.info(f"futility_stages: {futility_stages}")
        for key in range(1, 6):
            efficacy_all[str(key)] += efficacy_stages[str(key)]
            futility_all[str(key)] += futility_stages[str(key)]
            e_all[str(key)].append(efficacy_stages[str(key)])
            f_all[str(key)].append(futility_stages[str(key)])
        
        logger.info(f"lower bound: {robust_left}; evaluate PR: {Non_AE/1001};  epsilon: {epsilon}")
        logger.info(f"x,y: {x} {y} ")
        logger.info(f"--" * 50)  
    
    logger.info(f"efficacy_all: {efficacy_all}")
    logger.info(f"futility_all: {futility_all}")
    logger.info(f"e_all: {e_all}")
    logger.info(f"f_all: {f_all}")





