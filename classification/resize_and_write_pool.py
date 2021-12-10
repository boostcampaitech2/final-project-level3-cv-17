from genericpath import isdir, isfile
import cv2
import os
from os.path import join
import multiprocessing as mp

n_proc = os.cpu_count()//2
root = './validation_11'

def resize_and_write(folder):
    img_lst = os.listdir(join(root, folder))
    cnt = 0
    for im in img_lst:
        img_path = join(root, folder, im)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        dst = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_AREA)

        save_path = join(root, 'resize224', folder, im)
        
        if not os.path.exists(join(root, 'resize224', folder)):
            os.makedirs(join(root, 'resize224', folder))
        cv2.imwrite(save_path, dst)
        cnt+=1
        print(f'Resize {cnt} images in folder {folder}')

def main():
    folders = [folder for folder in os.listdir(root) if isdir(join(root, folder))]

    pool = mp.Pool(processes=n_proc)
    print(f'using {n_proc} processes')

    pool.map(resize_and_write, folders)
    pool.close()
    pool.join()

    print('done ')

if __name__ == '__main__':
    main()
        