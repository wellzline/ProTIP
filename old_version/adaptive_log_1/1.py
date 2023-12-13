from scipy.stats import shapiro
all = 0
count = 0
with open('log_word_1_20.log', 'r') as file:
    for line in file:
        if 'dis_loss' in line:
            group1 = [float(i) for i in line.strip().split('[')[1][:-1].split(',')]
            all += 1

            stat, p_value = shapiro(group1)
            print(f"Statistic: {stat}, p-value: {p_value}")
            if p_value > 0.05:
                count += 1
print("count:", count)
print("all:", all)
