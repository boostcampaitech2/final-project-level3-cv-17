import os
import random

root = '/opt/ml/final-project-level3-cv-17/classification/validaiton_11'

folder_list = os.listdir(root)       # ['02012001','02012002', ...]
for folder in folder_list:
    path = os.path.join(root,folder)
    file_list = os.listdir(path)    # ['14_142_14012002_160664755566085_0.jpg, ...,]
    # if len(file_list) <= 200:
    #     break
    save = []
    zero_cnt, one_cnt = 0,0
    for f in file_list:
        filename = f
        f = f.split(".")[0]
        if f[-1]=='0' and zero_cnt<20:
            save.append(filename)
            zero_cnt+=1
        elif f[-1]=='1' and one_cnt<20:
            save.append(filename)
            one_cnt+=1

    for f in file_list:
        if f in save:
            continue
        os.remove(os.path.join(path, f))
    # for filename in remove:
    #     os.remove(os.path.join(path, filename))