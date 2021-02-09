import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import sys
from scipy.ndimage.filters import gaussian_filter
import tps
from torchvision import transforms
import cv2

class Dataset(Dataset):
    
    def __init__(self, data_dir):
        self.img_list = [file for file in Path(data_dir).iterdir()]

        self.transformer_appearance = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transformer_sketch = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))])
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        gray_img = Image.open(self.img_list[index]).convert('L')
        img = np.array(img)
        gray_img = np.array(gray_img)
        
        sketch_img = xdog(gray_img)
        noise = np.random.uniform(-50, 50, img.shape)
        appearance_img = img+noise

        return self.transformer_appearance(appearance_img), self.transformer_sketch(sketch_img)

def xdog(image, epsilon=0.5, phi=10, k=1.4, tau=1, sigma=0.5):
    
    image = gaussian_filter(image, 0.7)
    gauss1 = gaussian_filter(image, sigma)
    gauss2 = gaussian_filter(image, sigma*k)

    D = gauss1 - tau*gauss2

    U = D/255
    
    for i in range(0,len(U)):
        for j in range(0,len(U[0])):
            U[i][j] = abs(1-U[i][j])
            
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if U[i][j] >= epsilon:
                U[i][j] = 1
            else:
                ht = np.tanh(phi*(U[i][j] - epsilon))
                U[i][j] = 1 + ht

    return U*255


