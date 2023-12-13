import Levenshtein
import re
import numpy as np

if __name__ == "__main__":
    for i in range(3,5):
        file_path = f"log_word_1_{i}0.log"
        dis_prompt = []
        with open(file_path, 'r') as file:
            for line in file:
                if 'dis_prompt' in line:
                    matches = re.findall(r'dis_prompt: (.+)', line)
                    for match in matches:
                        dis_prompt.append(match)
        print(len(dis_prompt))
        distance = []
        for item in dis_prompt[1:]:
            str1 = dis_prompt[0]
            str2 = item
            distance.append(Levenshtein.distance(str1, str2))
        mean_value = round(np.mean(distance), 2)
        print(f"编辑距离: {mean_value}")


for i in range(3,5):
    file_path = f"log_word_1_{i}0.log"
    score_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'dis_loss' in line:
                score = line.strip().split('[')[1][:-1].split(',')
                float_list = list(map(float, score))
                average_value = sum(float_list) / len(float_list)
                score_list.append(average_value)
    mean_value = round(np.mean(score_list), 2)
    print(f"AveSt2i: {mean_value}")