import os
from os.path import join
import torch
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *
import albumentations
from albumentations.pytorch.transforms import ToTensorV2


class BigDataset(Dataset):
    def __init__(self, data_dir, mode, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.image_paths = []
        self.food_labels = []
        self.data_dir = data_dir

        if mode == 'train':
            self.transforms = albumentations.Compose([
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.ToGray(p=1),
                    albumentations.CoarseDropout(max_holes=7, max_height=50, max_width=50, min_height=30, min_width=30, p=1),
                ], p=1),
                albumentations.OneOf([
                    albumentations.GaussNoise(p=1),
                    albumentations.GaussianBlur(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.HueSaturationValue(p=1),
                    albumentations.CLAHE(p=1),
                    albumentations.RandomGamma(p=1),
                    albumentations.ImageCompression(p=1),
                ], p=1),
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        elif mode == 'valid':
            self.transforms = albumentations.Compose([
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        self.setup()

    def setup(self):
        label_folders = sorted(os.listdir(self.data_dir))
        for cls, label_folder in enumerate(label_folders):      # deopbab ~ vegetable
            image_folders = sorted(os.listdir(join(self.data_dir, label_folder)))
            for image_folder in image_folders:
                image_path = join(self.data_dir, label_folder, image_folder)
                for filename in os.listdir(image_path):
                    self.image_paths.append(join(image_path, filename))
                    self.food_labels.append(cls)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        food_label = self.food_labels[index]

        image_transform = self.transforms(image=img)['image']
        return image_transform, food_label

    def __len__(self):
        return len(self.image_paths)

# custom = pd.read_csv('custom.csv')

class SmallDataset(Dataset):
    def __init__(self, data_dir, mode, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.image_paths = []
        self.food_labels = []
        self.data_dir = data_dir

        if mode == 'train':
            self.transforms = albumentations.Compose([
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.CoarseDropout(max_holes=7, max_height=50, max_width=50, min_height=30, min_width=30, p=1),
                ], p=1),
                albumentations.OneOf([
                    albumentations.GaussNoise(p=1),
                    albumentations.GaussianBlur(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.HueSaturationValue(p=1),
                    albumentations.CLAHE(p=1),
                    albumentations.RandomGamma(p=1),
                    albumentations.ImageCompression(p=1),
                ], p=1),
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        elif mode == 'valid':
            self.transforms = albumentations.Compose([
                albumentations.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        self.setup()

    def setup(self):
        image_folders = sorted(os.listdir(self.data_dir))
        for idx, image_folder in enumerate(image_folders):      # 01015002 ~ 08014003
            image_path = os.path.join(self.data_dir, image_folder)
            for filename in os.listdir(image_path):
                self.image_paths.append(os.path.join(image_path, filename))
                self.food_labels.append(idx)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        food_label = self.food_labels[index]

        image_transform = self.transforms(image=img)['image']
        return image_transform, food_label

    def __len__(self):
        return len(self.image_paths)
