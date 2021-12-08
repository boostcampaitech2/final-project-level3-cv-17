import os
import random

root = '/opt/ml/final-project-level3-cv-17/classification/validaiton_11'

folder_list = os.listdir(root)       # ['02012001','02012002', ...]
for folder in folder_list:
    path = os.path.join(root,folder)
    file_list = os.listdir(path)    # ['14_142_14012002_160664755566085_0.jpg, ...,]

    file_id = []        # ['160291732166069',...] (중복 O)
    for f in file_list:
        f = f.split(".")[0].split("_")
        if len(f) != 5:
            continue
        file_id.append(f[3])

    samples=[]      # 정위, 측위 사진 다 있는 id만 사용
    for k in file_id:
        if file_id.count(k) == 2:
            samples.append(k)
            file_id.remove(k)

    samples = random.sample(samples, 20)

    save=[]
    for f in file_list:
        file_name = f
        f = f.split(".")[0].split("_")
        if len(f) != 5:
            continue
        if (f[3] in samples[:100] and f[4]=="0") or (f[3] in samples[100:] and f[4] == "1"):
            save.append(file_name)

    for f in file_list:
        if f in save:
            continue
        os.remove(os.path.join(path, f))