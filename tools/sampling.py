import os
import random

path = '음식 이미지 및 영양정보 텍스트/Training/[원천]음식분류_TRAIN_048/train/15011001'

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

samples = random.sample(samples, 200)

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