# import os
# import json

# # process robustness value
# folders = ['./80/', './85/', './90/', './95/', './99/']
# folders = ['./80/', './95/']
# file_paths = []

# for item in folders:
#     file_paths.append(f"{item}log_char_{20}.log")

# print(len(file_paths), file_paths)
# with open('result.txt','w') as w_file:
#     for file_path in file_paths:
#         print("file_path:", file_path)
#         with open(file_path, 'r') as file:
#             for line in file:
#                 if "robust = [" in line:
#                     w_file.write(line)

import os
folders = ['./10/u-test/80/', './10/u-test/85/', './10/u-test/90/', './10/u-test/95/', './10/u-test/99/']
folders = ['./10/u-test/80/', './10/u-test/90/', './10/u-test/95/']

for index in range(1, 21):
    file_paths = []
    for item in folders:
        file_paths.append(f"{item}log_char_{index}.log")
    print(file_paths)

    print(len(file_paths), file_paths)
    with open(f"./result/10/u-test/{index}.txt",'w') as w_file:
        for file_path in file_paths:
            print("file_path:", file_path)
            with open(file_path, 'r') as file:
                for line in file:
                    if "robust = [" in line:
                        w_file.write(line)





# file_paths = []
# for index in range(1, 31):
#     file_paths.append(f"./20/u-test/95/log_char_{index}.log")

# print(len(file_paths), file_paths)
# with open('ealy_stop_20_95.txt','w') as w_file:
#     for file_path in file_paths:
#         stages = {'1':0, '2':0, '3':0, '4':0, '5':0}
#         print("file_path:", file_path)
#         with open(file_path, 'r') as file:
#             for line in file:
#                 if "Reject" in line:
#                     content = line.strip().split(':')[-1].strip()
#                     stages[content] += 1
#             json_data = json.dumps(stages)  

#             w_file.write(json_data + '\n')
# print(stages)



# with open('ealy_stop.txt', 'r') as json_file:
#     loaded_data = json.load(json_file)
# print(loaded_data)


