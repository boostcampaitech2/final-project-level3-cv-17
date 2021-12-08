import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *


class ClsDataset(Dataset):
    image_paths = []
    food_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.setup()

    def setup(self):
        image_folders = sorted(os.listdir(self.data_dir))
        for idx, image_folder in enumerate(image_folders):      # 11011001~11015002
            image_path = os.path.join(self.data_dir, image_folder)
            for filename in os.listdir(image_path):
                self.image_paths.append(os.path.join(image_path, filename))
                self.food_labels.append(idx)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = image.convert('RGB')
        food_label = self.food_labels[index]

        try:
            image_transform = self.transform(image)
        except:
            print('*'*50)
            print(self.image_paths[index])
            print('*'*50)
        return image_transform, food_label

    def __len__(self):
        return len(self.image_paths)


