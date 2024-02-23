
# s = ['autocorrect', 'spellchecker', 'gramfomer', "original"]
# robust_autocorrect, robust_spellchecker, robust_gramfomer, robust_original = [], [], [], []
# with open('SDV15_10_autocorrect.txt', 'r') as file_autocorrect, open('SDV15_10_spellchecker.txt', 'r') as file_spellchecker, \
#     open('SDV15_10_gramfomer.txt', 'r') as file_gramfomer, open('SDV15_10_original.txt', 'r') as file_original:

import csv

data = []
with open('SDV15_10_original.txt', 'r') as file:
    for line in file:
        robust = line.strip().split(';')[-2]
        data.append(float(robust))

# write into csv
with open('robustness_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for robustness in data:
        writer.writerow([robustness])
