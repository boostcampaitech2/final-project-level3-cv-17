import os
import shutil
import random

from os.path import join

root = 'train'
categories = os.listdir(root)               # deopbab ~ vegetable
sampled_lst = []

for category in categories:
    folders_path = join(root, category)     # ex) train/stew
    folders = os.listdir(folders_path)      # ex) 02011019 ~ 04019008
    for folder in folders:
        img_path = join(folders_path, folder)   # ex) train/stew/02011019
        img_list = os.listdir(img_path)
        samples = random.sample(img_list, len(img_list)//5)
        cnt = 0
        for sample in samples:
            path_fr = join(img_path, sample)    # ex) train/stew/02011019/02_...
            if not os.path.exists(join('valid', img_path)):
                os.makedirs(join('valid', img_path))
            path_to = join('valid', img_path, sample) # ex) valid/stew/02011019/02_...
            shutil.move(path_fr, path_to)
            cnt+=1
        print(f'move {cnt} files from {folder} folder to valid')