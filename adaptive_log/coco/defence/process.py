import os

# folders = ['./80/', './85/', './90/', './95/', './99/']
folders = ['./10/gramformer/', './10/spellchecker/', '../char/95/']
folders = ['./10/gramformer/', './10/spellchecker/', './10/autocorrect/','../char/10/u-test/95/']


for index in range(1, 37):
    file_paths = []
    for item in folders:
        file_paths.append(f"{item}log_char_{index}.log")
    print(file_paths)

    print(len(file_paths), file_paths)
    with open(f"./result/{index}.txt",'w') as w_file:
        for file_path in file_paths:
            print("file_path:", file_path)
            with open(file_path, 'r') as file:
                for line in file:
                    if "robust = [" in line:
                        w_file.write(line)



