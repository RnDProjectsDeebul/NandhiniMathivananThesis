import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from config import *
from sklearn.preprocessing import StandardScaler
import albumentations as A
import albumentations.augmentations.transforms as A_transform
from albumentations.pytorch import ToTensorV2
class ImageCoordinateDataset(Dataset):
    def __init__(self, root_dir, files, train_transform=None, valid_transform=None):
        self.root_dir = root_dir
        self.image_files = files
        self.scaler = StandardScaler()
        data = [[640, 480], [1, 1]]
        self.scaler.fit(data)
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        coordinate_file = image_file.replace('r.png', 'cpos.txt')
        coordinate_path = os.path.join(self.root_dir, coordinate_file)

        image = Image.open(image_path).convert('RGB')
        coordinate = self._load_coordinate(coordinate_path)

        if self.train_transform is not None:
            transformed = self.train_transform(image=np.array(image), keypoints=coordinate)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            transformed_keypoints = torch.tensor(transformed_keypoints, dtype=torch.float32)
            coordinates = self.scaler.transform(transformed_keypoints)
            image1 = np.transpose(transformed_image, (2, 0, 1)) 
        elif self.valid_transform is not None:
            transformed = self.valid_transform(image=np.array(image))
            transformed_image = transformed['image']
            coordinates = self.scaler.transform(coordinate)
            image1 = np.transpose(transformed_image, (2, 0, 1)) 
        else:
            image1 = np.array(image)
            coordinates = self.scaler.transform(coordinate)
            image1 = image1.transpose((2, 0, 1))

        return image1, coordinates
    
    def _load_coordinate(self, coordinate_path):
        with open(coordinate_path, 'r') as f:
            lines = f.readlines()
        
        coordinates = []
        
        for line in lines[:8]:
            x, y = map(float, line.strip().split())
            coordinates.append([x, y])
            
        return torch.tensor(coordinates)