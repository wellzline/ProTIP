
s = ['autocorrect', 'spellchecker', 'gramfomer', "original"]
robust_autocorrect, robust_spellchecker, robust_gramfomer, robust_original = [], [], [], []
with open('SDV15_10_autocorrect.txt', 'r') as file_autocorrect, open('SDV15_10_spellchecker.txt', 'r') as file_spellchecker, \
    open('SDV15_10_gramfomer.txt', 'r') as file_gramfomer, open('SDV15_10_original.txt', 'r') as file_original:
    robust_autocorrect = file_autocorrect.readlines()
    robust_spellchecker = file_spellchecker.readlines()
    robust_gramfomer = file_gramfomer.readlines()
    robust_original = file_original.readlines()

for index in range(1, 37):
    with open(f"./draw_result/{index}.txt", 'w') as file:
        file.write(robust_autocorrect[index - 1][:-2] + '\n')
        file.write(robust_spellchecker[index - 1][:-2] + '\n')
        file.write(robust_gramfomer[index - 1][:-2] + '\n')
        file.write(robust_original[index - 1][:-2] + '\n') 
