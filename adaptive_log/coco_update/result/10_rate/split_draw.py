

robust_70, robust_80, robust_90 = [], [], []
with open('SDV15_10_70.txt', 'r') as file_70, open('SDV15_10_80.txt', 'r') as file_80, open('SDV15_10_90.txt', 'r') as file_90:
    robust_70 = file_70.readlines()
    robust_80 = file_80.readlines()
    robust_90 = file_90.readlines()

for index in range(1, 37):
    with open(f"./draw_result/{index}.txt", 'w') as file:
        file.write(robust_70[index - 1][:-2] + '\n')
        file.write(robust_80[index - 1][:-2] + '\n')
        file.write(robust_90[index - 1][:-2] + '\n')
