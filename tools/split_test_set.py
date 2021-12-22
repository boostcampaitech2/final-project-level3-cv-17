import os
import random
import argparse
import shutil
from tqdm import tqdm


# 대분류 폴더 파악 후 데이터 옮기기 시작
def split_test_dir(train_dir:str, split_amount:int):
    data_dir = os.path.dirname(train_dir.rstrip('/'))
    move_fail_list = []

    for custom_class in os.listdir(train_dir):
        for folder in tqdm(os.listdir(os.path.join(train_dir, custom_class))):
            path = os.path.join(train_dir, custom_class, folder)
            new_path = os.path.join(data_dir, 'test', custom_class, folder)
            split_data(path, new_path, split_amount)

            move_fail = check_file_amount(new_path, split_amount)

            if move_fail:
                move_fail_list.append(move_fail)

    # 파일이 덜 옮겨진 폴더 있으면 출력
    if move_fail_list:
        print("There's still left to move.")
        print('(folder path, file amount)')
        print(*move_fail_list, sep='\n')


# 각 소분류 폴더 내 이미지 이동
def split_data(path:str, new_path:str, split_amount:int):
    image_list = os.listdir(path)
    image_list = random.sample(image_list, split_amount)
    
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for il in image_list:
        shutil.move(os.path.join(path, il), new_path)


# 설정한 양만큼 이동했는지 체크
def check_file_amount(new_path:str, split_amount:int):
    file_amount = len(os.listdir(new_path))
    if file_amount != split_amount:
        return (new_path, file_amount)


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='../data/train')
parser.add_argument('--backup', type=lambda arg: arg.lower() in ['true', 't', '1'], default=True)
parser.add_argument('--split_amount', type=int, default=100)
args = parser.parse_args()

train_dir = args.train_dir
backup = args.backup
split_amount = args.split_amount

try:
    # 기존 data backup
    if backup:
        # TODO Progress bar
        print('Start backup')
        shutil.copytree(train_dir, f'{train_dir}_backup')
        print(f'{train_dir} backup complete')

    # test set 분할
    split_test_dir(train_dir, split_amount)

except FileNotFoundError as fe:
    print("!!!!!!!!!!!!!!!!!")
    print("Can't find val directory. Please Check your path")
    print(f"Input path: {train_dir}")
    print("!!!!!!!!!!!!!!!!!")