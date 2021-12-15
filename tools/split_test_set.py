import os
import random
import argparse
import shutil


def split_test_dir(train_dir:str):
    data_dir = os.path.dirname(train_dir.rstrip('/'))
    for custom_class in os.listdir(train_dir):
        for folder in os.listdir(os.path.join(train_dir, custom_class)):
            path = os.path.join(train_dir, custom_class, folder)
            new_path = os.path.join(data_dir, 'test', custom_class, folder)
            split_data(path, new_path)


def split_data(path:str, new_path:str):
    image_list = os.listdir(path)
    image_list = random.sample(image_list, 100)
    
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for il in image_list:
        shutil.move(os.path.join(path, il), new_path)

    print(f'Moving 100 files from {path} to {new_path} complete')


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='../data/train')
args = parser.parse_args()

train_dir = args.train_dir

try:
    shutil.copytree(train_dir, f'{train_dir}_backup')   # 기존 data backup
    split_test_dir(train_dir)
except FileNotFoundError as fe:
    print("!!!!!!!!!!!!!!!!!")
    print("Can't find val directory. Please Check your path")
    print(f"Input path: {train_dir}")
    print("!!!!!!!!!!!!!!!!!")