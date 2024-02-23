import logging

def setup_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger 

logger = setup_logger(f"./result/all_result_5.log")

def calculate_R(E_n, n):
    import math
    sigma = 0.1
    robust_left, robust_right, epsilon = 0, 0, 0
    epsilon = math.sqrt( (0.6 * math.log(math.log(n, 1.1) + 1, 10) + (1.8 ** -1) * math.log(24/sigma, 10)) / n )
    robust_left = E_n/n - epsilon
    robust_right = E_n/n + epsilon
    return robust_left, robust_right, epsilon

e_all = {'1':[], '2':[], '3':[], '4':[], '5':[]}
f_all = {'1':[], '2':[], '3':[], '4':[], '5':[]}
efficacy_all = {'1':0, '2':0, '3':0, '4':0, '5':0}
futility_all = {'1':0, '2':0, '3':0, '4':0, '5':0}

robust = []
for index in range(31, 32):
    # if index != 5:
    with open(f"./SDV2/90/log_char_{index}.log", 'r') as file, open('result/efficitive.txt', 'a+') as file_w:
        content = file.readlines()
        for line in content:
            if "robust reach" in line:
                robust.append(line.strip().split(':')[-1].split(',')[0].strip())
        file_w.write(';'.join(i for i in robust) + '\n')
    
    logger.info(f"--" * 50) 
    logger.info(f"prompt: {index}") 

    alpha = [0.0148, 0.0262, 0.0354, 0.0432, 0.05]
    futility_boundary = [-0.145, 0.511, 1.027, 1.497, float('inf')]
    efficacy_boundary = [2.176, 2.144, 2.113, 2.090, 2.071]  # critical value when Z-score > critical value, the reject the H0
            
    efficacy_stages = {'1':0, '2':0, '3':0, '4':0, '5':0}
    futility_stages = {'1':0, '2':0, '3':0, '4':0, '5':0}

    p_val, z_val = 0, 0
    stage = {'12':1, '24':2, '36':3, '48':4, '60':5}
    Non_AE, AE = 0, 0
    n = 0
    for i in range(len(content)):
        if "stop_early: 0" in content[i]:
            n += 1
            line = content[i-1].strip().split(':')[-1].split()  # "p_val, z_val" 
            p_val, z_val = float(line[0]), float(line[1])
            s = content[i-2].split(';')[0].split(':')[-1].strip()

            if z_val >= efficacy_boundary[stage[s]-1]:
                AE += 1
                efficacy_stages[str(stage[s])] += 1
                continue

            if z_val <= futility_boundary[stage[s]-1]:
                Non_AE += 1
                futility_stages[str(stage[s])] += 1
                continue
            # if s == '60':
            #     if p_val > alpha[stage[s]-1]:
            #         Non_AE += 1
            #         futility_stages[str(stage[s])] += 1
            #     else:
            #         AE += 1
            #         efficacy_stages[str(stage[s])] += 1
        if n >= 1:
            robust_left, robust_right, epsilon = calculate_R(Non_AE, n)
    logger.info(f"AE: {AE}")
    logger.info(f"Non_AE: {Non_AE}")
    logger.info(f"n: {n}")
    logger.info(f"lower bound: {robust_left}; evaluate PR: {Non_AE/n};  epsilon: {epsilon}")
    logger.info(f"efficacy_stages: {efficacy_stages}")
    logger.info(f"futility_stages: {futility_stages}")
    logger.info(f"--" * 50)  

    for key in range(1, 6):
        efficacy_all[str(key)] += efficacy_stages[str(key)]
        futility_all[str(key)] += futility_stages[str(key)]
        e_all[str(key)].append(efficacy_stages[str(key)])
        f_all[str(key)].append(futility_stages[str(key)])


logger.info(f"efficacy_all: {efficacy_all}")
logger.info(f"futility_all: {futility_all}")
logger.info(f"e_all: {e_all}")
logger.info(f"f_all: {f_all}")


