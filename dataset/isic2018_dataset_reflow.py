import os
import glob
import json
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
# import albumentations as A
from sklearn.model_selection import KFold


def norm01(x):
    return np.clip(x, 0, 255) / 255


seperable_indexes = json.load(open('/root/volume/dataset/ISIC2018/data_split.json', 'r'))
# cross validation
class ISICDataset(data.Dataset):
    def __init__(self, root_dir, fold, split, reflow_dir, transform=False):
        super(ISICDataset, self).__init__()
        self.split = split
        # root_dir = '/root/volume/dataset/ISIC2018'

        # load images, label, point
        self.image_paths = []
        self.label_paths = []
        self.dist_paths  = []

        indexes = [l[5:-4] for l in os.listdir(root_dir + '/Data/')]
        valid_indexes = seperable_indexes[fold]
        train_indexes = list(filter(lambda x: x not in valid_indexes, indexes))
        print('Fold {}: train: {} valid: {}'.format(fold, len(train_indexes),
                                                    len(valid_indexes)))
        indexes = train_indexes if split == 'train' else valid_indexes

        self.image_paths = [
            root_dir + '/Data/ISIC_{}.jpg'.format(_id) for _id in indexes
        ]
        self.label_paths = [
            root_dir + '/GroundTruth/ISIC_{}_segmentation.png'.format(_id) for _id in indexes
        ]
        self.noise_paths = [
            reflow_dir + '/ISIC_{}.npy'.format(_id) for _id in indexes
        ]
        

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.transform = transform

        # p = 0.5
        # self.transf = A.Compose([
        #     A.GaussNoise(p=p),
        #     A.HorizontalFlip(p=p),
        #     A.VerticalFlip(p=p),
        #     A.ShiftScaleRotate(p=p),
        #     # A.RandomBrightnessContrast(p=p),
        # ])
    def __getitem__(self, index):
        img   = Image.open(self.image_paths[index]).convert('RGB')
        mask  = Image.open(self.label_paths[index]).convert('L')
        noise = np.load(self.noise_paths[index]) # pair noise
        noise = torch.from_numpy(noise).float()
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform["image"](img)
            torch.set_rng_state(state)
            mask = self.transform["mask"](mask)
            
        return (img, mask, noise)

    def __len__(self):
        return self.num_samples
