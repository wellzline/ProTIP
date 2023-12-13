import logging
import time
logging.basicConfig(
    filename='log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
cosine_group = []
p_val = []
p_prob = []
with open('log_insertion.log', 'r') as file:
    for line in file:
        if 'p_val' in line:
            p_val.append(float(line.strip().split()[-1]))
        if "cosine_group" in line:
            cosine_group.append(int(line.strip().split()[7].strip(',')))
        if "prob" in line:
            p_prob.append(float(line.strip().split()[-1]))

print(len(p_val))
print(len(cosine_group))
print(len(p_prob))

logging.info(f"x =  {p_val}")
logging.info(f"y =  {p_prob}")
logging.info(f"z =  {cosine_group}")
