import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

class ISICDataset(Dataset):
    def __init__(self, data_path, df, transform = None, mode = 'Training'):
        # Validationの名前のフォルダはなく、大元がTrainingのフォルダになっている
        if mode == "Validation":
            mode = "Training"
        self.name_list   = df.iloc[:, 0].tolist()  # 1番目の列が画像のファイル名（ISIC_0000000 形式）
        self.data_path   = os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_Data')
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name + '.jpg')  # 画像ファイルのパス
        msk_path = os.path.join(self.data_path+'_GroundTruth', name + '_Segmentation.png')  # セグメンテーションマスクのパス


        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform["image"](img)
            torch.set_rng_state(state)
            mask = self.transform["mask"](mask)
            
        return (img, mask)